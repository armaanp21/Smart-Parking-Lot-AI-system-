import cv2
import depthai as dai
import numpy as np
import json
import boto3
import time
import threading
from flask import Flask, Response, request, jsonify
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/parking_model.blob"
CONFIG_FILE = "parking_config.json"
DYNAMO_TABLE_NAME = "ParkingData"
AWS_REGION = "us-east-1"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45  # Overlap threshold for filtering

# --- AWS SETUP ---
try:
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMO_TABLE_NAME)
    print("Connected to AWS DynamoDB")
except Exception as e:
    print(f"AWS Connection Failed: {e}")

# --- FLASK SETUP ---
app = Flask(__name__)
frame_for_web = None
lock = threading.Lock()

# --- HELPER: YOLOv8 DECODING ---
def decode_yolov8(output_layer, conf_thresh, iou_thresh):
    # YOLOv8 output is usually shape (1, 4+Classes, 8400) -> e.g. (1, 6, 8400)
    # We flatten it to 1D then reshape
    data = np.array(output_layer).reshape(6, 8400) # 6 = x,y,w,h + 2 classes
    
    # Transpose to (8400, 6) so each row is a box
    data = data.transpose()
    
    # Extract Class 1 (Occupied) Scores. (Index 5 because: 0=x, 1=y, 2=w, 3=h, 4=empty, 5=occupied)
    # If your classes are flipped, change to data[:, 4]
    scores = data[:, 5] 
    
    # Filter out weak detections
    mask = scores > conf_thresh
    if not np.any(mask):
        return []
        
    filtered_data = data[mask]
    filtered_scores = scores[mask]
    
    # Prepare boxes for NMS (Center X, Center Y, W, H) -> (Top, Left, W, H)
    boxes = filtered_data[:, :4]
    
    # YOLOv8 outputs pixels (0-640). We need OpenCV format (x, y, w, h)
    # Convert cx,cy,w,h to x,y,w,h (Top-Left)
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2) # x = cx - w/2
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2) # y = cy - h/2
    
    # Apply Non-Maximum Suppression (Remove duplicates)
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(), 
        scores=filtered_scores.tolist(), 
        score_threshold=conf_thresh, 
        nms_threshold=iou_thresh
    )
    
    final_detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            # Normalize to 0-1 range for our existing logic
            b = boxes[i]
            x_norm = b[0] / 640.0
            y_norm = b[1] / 640.0
            w_norm = b[2] / 640.0
            h_norm = b[3] / 640.0
            
            final_detections.append([x_norm, y_norm, w_norm, h_norm])
            
    return final_detections

# --- OAK-D PIPELINE ---
def create_pipeline():
    pipeline = dai.Pipeline()

    # Camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(640, 640)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # RAW NEURAL NETWORK (Not YoloDetectionNetwork)
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(MODEL_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    
    # Outputs
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("nn")

    camRgb.preview.link(nn.input)
    camRgb.preview.link(xoutRgb.input)
    nn.out.link(xoutNN.input)

    return pipeline

# --- MAIN LOGIC ---
def run_oak_d_thread():
    global frame_for_web
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    pipeline = create_pipeline()
    
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        
        last_aws_update = 0

        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            
            frame = inRgb.getCvFrame()
            
            # --- CUSTOM YOLOv8 PARSING ---
            # Get raw layer data
            output_layer = inDet.getFirstLayerFp16() 
            # Decode it using our helper function
            detections = decode_yolov8(output_layer, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

            # Resize to 640x480 for Web UI
            frame_resized = cv2.resize(frame, (640, 480))

            # Load Mapping Config
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except:
                config = {"spots": [], "entrance": None, "rate": 0.05}

            current_status_batch = []
            
            if config['spots']:
                for spot in config['spots']:
                    is_occupied = False
                    
                    # Check overlap with our decoded boxes
                    for det in detections:
                        # det is [x, y, w, h] normalized 0-1
                        x1 = int(det[0] * 640)
                        y1 = int(det[1] * 480)
                        w = int(det[2] * 640)
                        h = int(det[3] * 480)
                        cx, cy = x1 + w//2, y1 + h//2
                        
                        # Check if car center is inside parking spot
                        if (spot['x'] < cx < spot['x'] + spot['w'] and 
                            spot['y'] < cy < spot['y'] + spot['h']):
                            is_occupied = True
                            break
                    
                    # Draw Visuals
                    color = (0, 0, 255) if is_occupied else (0, 255, 0)
                    cv2.rectangle(frame_resized, (int(spot['x']), int(spot['y'])), 
                                  (int(spot['x']+spot['w']), int(spot['y']+spot['h'])), color, 2)
                    
                    # Distance Calc
                    dist = 0
                    if config['entrance']:
                        dist_px = np.sqrt((spot['x'] - config['entrance']['x'])**2 + 
                                          (spot['y'] - config['entrance']['y'])**2)
                        dist = round(dist_px * 0.05, 1)

                    current_status_batch.append({
                        'SpotID': spot['id'],
                        'Status': 'Occupied' if is_occupied else 'Available',
                        'Distance': dist, # Number type for GSI sorting
                        'Rate': config.get('rate', 0.05)
                    })

                # UPDATE AWS (Every 3 seconds)
                if time.time() - last_aws_update > 3:
                    threading.Thread(target=update_aws, args=(current_status_batch,)).start()
                    last_aws_update = time.time()

            ret, buffer = cv2.imencode('.jpg', frame_resized)
            with lock:
                frame_for_web = buffer.tobytes()
            
            time.sleep(0.01)

def update_aws(data):
    try:
        with table.batch_writer() as batch:
            for item in data:
                # Need to convert float to Decimal for DynamoDB
                # But boto3 handles float -> Decimal automatically often, 
                # if not we might need a helper. For now let's try raw.
                # If error, we cast to Decimal.
                item['Rate'] =  str(item['Rate']) # Safety: Dynamo sometimes hates floats
                item['Distance'] = int(item['Distance']) # Simplify to Int for safety
                batch.put_item(Item=item)
        print("AWS Updated")
    except Exception as e:
        print(f"AWS Update Failed: {e}")

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_config')
def get_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify(None)

@app.route('/save_config', methods=['POST'])
def save_config():
    with open(CONFIG_FILE, 'w') as f:
        json.dump(request.json, f)
    return jsonify({"status": "success"})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if frame_for_web:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_for_web + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=run_oak_d_thread)
    t.daemon = True
    t.start()
    
    print("Server starting... Access at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)