import streamlit as st
import cv2
import depthai as dai
import numpy as np
import json
import boto3
import time
import threading
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from decimal import Decimal

# --- CONFIGURATION ---
MODEL_PATH = "models/parking_model.blob"
CONFIG_FILE = "parking_config.json"
DYNAMO_TABLE_NAME = "ParkingData"
AWS_REGION = "us-east-1"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# --- HELPER: YOLOv8 DECODER (The Math) ---
def decode_yolov8(output_layer):
    # 1. Reshape 8400 outputs
    data = np.array(output_layer).reshape(6, 8400).transpose()
    
    # 2. Filter by Class 1 (Occupied Space) & Confidence
    # Note: data[4] is usually class 0, data[5] is class 1. 
    # Adjust index '5' if your 'occupied' class ID is different.
    scores = data[:, 5] 
    mask = scores > CONFIDENCE_THRESHOLD
    if not np.any(mask): return []
    
    filtered_data = data[mask]
    filtered_scores = scores[mask]
    boxes = filtered_data[:, :4]
    
    # 3. Convert Center-X/Y to Top-Left X/Y for OpenCV
    # Boxes are [cx, cy, w, h] -> need [x, y, w, h]
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    
    # 4. NMS (Non-Maximum Suppression) to remove duplicates
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(), 
        scores=filtered_scores.tolist(), 
        score_threshold=CONFIDENCE_THRESHOLD, 
        nms_threshold=IOU_THRESHOLD
    )
    
    final_detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            b = boxes[i]
            # Normalize to 0-1 range
            final_detections.append([b[0]/640.0, b[1]/640.0, b[2]/640.0, b[3]/640.0])
            
    return final_detections

# --- SYSTEM CLASS ---
@st.cache_resource
class ParkingSystem:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.config = self.load_config()
        self.running = True
        self.detections = [] # Store latest car detections
        self.last_aws_time = 0
        self.aws_status = "Waiting..."
        
        # Connect AWS
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
            self.table = self.dynamodb.Table(DYNAMO_TABLE_NAME)
            print("AWS Connected")
        except:
            print("AWS Connection Failed")

        # Start Camera Thread
        self.thread = threading.Thread(target=self.run_oak_d)
        self.thread.daemon = True
        self.thread.start()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"spots": [], "entrance": {"x": 320, "y": 480}, "rate": 0.05}

    def save_config(self, new_config):
        self.config = new_config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_config, f)

    def run_oak_d(self):
        pipeline = dai.Pipeline()
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(MODEL_PATH)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("nn")

        camRgb.preview.link(nn.input)
        camRgb.preview.link(xoutRgb.input)
        nn.out.link(xoutNN.input)

        with dai.Device(pipeline) as device:
            qRgb = device.getOutputQueue("rgb", 4, False)
            qDet = device.getOutputQueue("nn", 4, False)
            
            while self.running:
                inRgb = qRgb.get()
                inDet = qDet.get()
                
                # 1. Get Frame
                frame_cv = inRgb.getCvFrame()
                frame_resized = cv2.resize(frame_cv, (640, 480))
                
                # 2. Get Detections (Real Math)
                layer = inDet.getFirstLayerFp16()
                self.detections = decode_yolov8(layer)
                
                # 3. Check Spots Intersection
                status_batch = []
                if self.config['spots']:
                    for spot in self.config['spots']:
                        is_occupied = False
                        
                        # Check if any car center is inside this spot
                        for det in self.detections:
                            # det is [x,y,w,h] normalized
                            cx = (det[0] * 640) + (det[2] * 640 / 2)
                            cy = (det[1] * 480) + (det[3] * 480 / 2)
                            
                            # Simple "Center in Box" check
                            if (spot['x'] < cx < spot['x'] + spot['w'] and 
                                spot['y'] < cy < spot['y'] + spot['h']):
                                is_occupied = True
                                break
                        
                        # Calculate Distance
                        dist = 0
                        if self.config.get('entrance'):
                            dist = np.sqrt((spot['x'] - self.config['entrance']['x'])**2 + 
                                           (spot['y'] - self.config['entrance']['y'])**2)
                            dist = int(dist * 0.05) # Convert pixels to meters (approx)

                        status_batch.append({
                            'SpotID': spot['id'],
                            'Status': 'Occupied' if is_occupied else 'Available',
                            'Distance': Decimal(str(dist)), # Use Decimal for DynamoDB
                            'Rate': Decimal(str(self.config.get('rate', 0.05)))
                        })

                # 4. AWS Update (Every 3s)
                if time.time() - self.last_aws_time > 3 and status_batch:
                    threading.Thread(target=self.update_aws, args=(status_batch,)).start()
                    self.last_aws_time = time.time()

                with self.lock:
                    self.frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                time.sleep(0.01)

    def update_aws(self, data):
        try:
            with self.table.batch_writer() as batch:
                for item in data:
                    batch.put_item(Item=item)
            self.aws_status = f"Last Update: {time.strftime('%H:%M:%S')}"
        except Exception as e:
            self.aws_status = f"AWS Error: {str(e)[:20]}"

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# --- APP UI START ---
system = ParkingSystem()

st.set_page_config(page_title="Parking AI", layout="wide")
st.title("Smart Parking System")

# Status Bar
col1, col2 = st.columns(2)
col1.metric("System Status", "Running")
col2.metric("DynamoDB Sync", system.aws_status)

mode = st.sidebar.radio("Operation Mode", ["Live Monitor", "Setup / Calibration"])

if mode == "Setup / Calibration":
    st.info("Instructions: 1. Draw boxes on spots. 2. Click Save. (Double-click box to delete)")
    frame = system.get_frame()
    
    if frame is not None:
        bg_img = Image.fromarray(frame)
        
        # Load existing spots into canvas
        init_draw = {"objects": []}
        if "spots" in system.config:
            for s in system.config['spots']:
                init_draw["objects"].append({
                    "type": "rect", "left": s['x'], "top": s['y'], 
                    "width": s['w'], "height": s['h'],
                    "stroke": "#00FF00", "strokeWidth": 2, "fill": "rgba(0,255,0,0.2)"
                })
        
        # Drawing Canvas
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=2, stroke_color="#00FF00",
            background_image=bg_img,
            height=480, width=640,
            drawing_mode="rect",
            initial_drawing=init_draw,
            key="canvas_parking"
        )
        
        if st.button("Save Configuration"):
            if canvas.json_data:
                new_spots = []
                for i, obj in enumerate(canvas.json_data["objects"]):
                    new_spots.append({
                        "id": f"A{i+1}",
                        "x": int(obj["left"]), "y": int(obj["top"]),
                        "w": int(obj["width"]), "h": int(obj["height"])
                    })
                cfg = system.config
                cfg["spots"] = new_spots
                system.save_config(cfg)
                st.success(f"Saved {len(new_spots)} spots! Reloading...")
                time.sleep(1)
                st.rerun()

else:
    # LIVE MONITOR
    show_debug = st.sidebar.checkbox("Show AI Detections (Debug)", value=True)
    
    placeholder = st.empty()
    while True:
        frame = system.get_frame()
        if frame is not None:
            # Draw Overlays on top of the clean frame
            # 1. Draw Parking Spots (Green/Red)
            for spot in system.config.get('spots', []):
                # We need to re-check occupancy here just for visual drawing
                # (Or access the thread's last status if we stored it)
                color = (0, 255, 0) # Green default
                
                # Check occupancy for color
                for det in system.detections:
                    cx = (det[0] * 640) + (det[2] * 640 / 2)
                    cy = (det[1] * 480) + (det[3] * 480 / 2)
                    if (spot['x'] < cx < spot['x'] + spot['w'] and 
                        spot['y'] < cy < spot['y'] + spot['h']):
                        color = (255, 0, 0) # Red if occupied
                        break
                
                cv2.rectangle(frame, (spot['x'], spot['y']), 
                             (spot['x']+spot['w'], spot['y']+spot['h']), color, 2)
                cv2.putText(frame, spot['id'], (spot['x'], spot['y']-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 2. Draw Raw AI Detections (If Debug is ON)
            if show_debug:
                for det in system.detections:
                    # det is [x,y,w,h]
                    x = int(det[0] * 640)
                    y = int(det[1] * 480)
                    w = int(det[2] * 640)
                    h = int(det[3] * 480)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.putText(frame, "CAR", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            placeholder.image(frame)
        time.sleep(0.1)