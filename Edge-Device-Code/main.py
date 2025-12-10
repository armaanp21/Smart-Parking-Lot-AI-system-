import cv2
import depthai as dai
import numpy as np
import json
import boto3
import time
import threading
from flask import Flask, render_template, Response, request, jsonify
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/parking_model.blob"  
CONFIG_FILE = "parking_config.json"
DYNAMO_TABLE_NAME = "ParkingData"         
AWS_REGION = "ca-central-1"                  

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

# --- OAK-D PIPELINE ---
def create_pipeline():
    pipeline = dai.Pipeline()

    # Camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(640, 640)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # YOLO Detection Network
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork.setBlobPath(MODEL_PATH)
    detectionNetwork.setConfidenceThreshold(0.5)
    
    # YOLO Settings (Adjust based on your training)
    detectionNetwork.setNumClasses(2)  
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]) 
    detectionNetwork.setAnchorMasks({"side26": [1,2,3], "side13": [3,4,5]})
    detectionNetwork.setInputSize(640, 640)

    # Outputs
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("nn")

    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(xoutRgb.input)
    detectionNetwork.out.link(xoutNN.input)

    return pipeline

# --- MAIN LOGIC ---
def run_oak_d_thread():
    global frame_for_web
    
    # Retry finding model if not immediately available
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
            detections = inDet.detections

            # Resize to match our Web UI (640x480)
            frame_resized = cv2.resize(frame, (640, 480))

            # Load Mapping Config
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except:
                config = {"spots": [], "entrance": None, "rate": 0.05}

            # PROCESS SPOTS
            if config['spots']:
                for spot in config['spots']:
                    is_occupied = False
                    
                    # Check overlap
                    for det in detections:
                        # --- NEW CODE START ---
                        # REPLACE '1' WITH THE CLASS ID FOR "SPACE-OCCUPIED" FROM YOUR YAML FILE
                        OCCUPIED_CLASS_ID = 1 
                        
                        if det.label != OCCUPIED_CLASS_ID:
                            continue # Skip "empty" detections
                        # --- NEW CODE END ---

                        # Normalize detection to 640x480
                        x1 = int(det.xmin * 640)
                        y1 = int(det.ymin * 480)
                        x2 = int(det.xmax * 640)
                        y2 = int(det.ymax * 480)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Check if detection center is inside spot box
                        if (spot['x'] < cx < spot['x'] + spot['w'] and 
                            spot['y'] < cy < spot['y'] + spot['h']):
                            is_occupied = True
                            break
                    
                    # Visual Feedback
                    color = (0, 0, 255) if is_occupied else (0, 255, 0)
                    cv2.rectangle(frame_resized, (int(spot['x']), int(spot['y'])), 
                                  (int(spot['x']+spot['w']), int(spot['y']+spot['h'])), color, 2)
                    
                    # Calculate Distance (Approx using pixels if depth not avail)
                    dist = 0
                    if config['entrance']:
                        dist_px = np.sqrt((spot['x'] - config['entrance']['x'])**2 + 
                                          (spot['y'] - config['entrance']['y'])**2)
                        dist = round(dist_px * 0.05, 1) # Dummy calibration factor (100px = 5m)

                    current_status_batch.append({
                        'SpotID': spot['id'],
                        'Status': 'Occupied' if is_occupied else 'Available',
                        'Distance': str(dist), # Storing as string or number depends on your Dynamo setup
                        'Rate': str(config.get('rate', 0.05))
                    })

                # UPDATE AWS (Every 3 seconds to save bandwidth)
                if time.time() - last_aws_update > 3:
                    threading.Thread(target=update_aws, args=(current_status_batch,)).start()
                    last_aws_update = time.time()

            # Encode for Web Stream
            ret, buffer = cv2.imencode('.jpg', frame_resized)
            with lock:
                frame_for_web = buffer.tobytes()

def update_aws(data):
    try:
        with table.batch_writer() as batch:
            for item in data:
                batch.put_item(Item=item)
        print(" AWS Updated")
    except Exception as e:
        print(f" AWS Update Failed: {e}")

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

# --- ENTRY POINT ---
if __name__ == '__main__':
    # Start Camera Thread
    t = threading.Thread(target=run_oak_d_thread)
    t.daemon = True
    t.start()
    
    # Start Web Server (Accessible on local network)
    print("Server starting... Access at http://<RASPBERRY_PI_IP>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)