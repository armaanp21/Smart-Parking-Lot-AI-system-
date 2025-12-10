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

# --- CONFIGURATION ---
MODEL_PATH = "models/parking_model.blob"
CONFIG_FILE = "parking_config.json"
DYNAMO_TABLE_NAME = "ParkingData"
AWS_REGION = "us-east-1"

# --- CACHED BACKGROUND SYSTEM ---
# This ensures the camera and AWS connection run once and don't restart when you click buttons.
@st.cache_resource
class ParkingSystem:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.config = self.load_config()
        self.running = True
        self.detections = []
        
        # Connect AWS
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
            self.table = self.dynamodb.Table(DYNAMO_TABLE_NAME)
            print("✅ AWS Connected")
        except:
            print("⚠️ AWS Connection Failed")

        # Start Camera Thread
        self.thread = threading.Thread(target=self.run_oak_d)
        self.thread.daemon = True
        self.thread.start()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {"spots": [], "entrance": {"x": 320, "y": 480}, "rate": 0.05}

    def save_config(self, new_config):
        self.config = new_config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_config, f)
        print("💾 Configuration Saved")

    def run_oak_d(self):
        # 1. Pipeline Setup (YOLOv8 Logic)
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

        # 2. Loop
        with dai.Device(pipeline) as device:
            qRgb = device.getOutputQueue("rgb", 4, False)
            qDet = device.getOutputQueue("nn", 4, False)
            last_aws_update = 0

            while self.running:
                inRgb = qRgb.get()
                inDet = qDet.get()
                frame_cv = inRgb.getCvFrame()
                
                # Resize to 640x480 for UI consistency
                frame_resized = cv2.resize(frame_cv, (640, 480))
                
                # Simple Detection Processing (Simplified for speed)
                # Note: For production, re-insert the decode_yolov8 logic here.
                # For Setup UI, we just need the frame mostly.
                output_layer = inDet.getFirstLayerFp16()
                # (Assuming you add the decode_yolov8 function back here or import it)
                # self.detections = decode_yolov8(...) 

                # Process Spots Logic
                status_batch = []
                if self.config['spots']:
                    for spot in self.config['spots']:
                        # Simulate check for demo (Replace with real overlap logic)
                        # Here we just draw them on the frame for the 'Live View'
                        color = (0, 255, 0)
                        cv2.rectangle(frame_resized, (spot['x'], spot['y']), 
                                     (spot['x']+spot['w'], spot['y']+spot['h']), color, 2)
                        
                        # Add Logic to detect occupancy here
                        # ...
                        
                        status_batch.append({
                            'SpotID': spot['id'],
                            'Status': 'Available', # Update this based on logic
                            'Distance': int(10),   # Update this based on logic
                            'Rate': str(self.config.get('rate', 0.05))
                        })

                # AWS Update (Throttle 3s)
                if time.time() - last_aws_update > 3:
                    # threading.Thread(target=self.update_aws, args=(status_batch,)).start()
                    last_aws_update = time.time()

                # Update Global Frame
                with self.lock:
                    self.frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                time.sleep(0.01) # Important for Streamlit

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# Initialize System
system = ParkingSystem()

# --- STREAMLIT UI ---
st.title("🅿️ Smart Parking Manager")

# Sidebar Controls
mode = st.sidebar.radio("Mode", ["Live Monitor", "Setup / Calibration"])

if mode == "Setup / Calibration":
    st.header("Calibration Mode")
    st.info("Draw boxes around parking spots. Double-click a box to remove it.")
    
    # Get one static frame for drawing
    frame = system.get_frame()
    if frame is not None:
        bg_image = Image.fromarray(frame)
        
        # Load existing spots into canvas
        initial_drawing = {"objects": []}
        for spot in system.config['spots']:
            initial_drawing["objects"].append({
                "type": "rect",
                "left": spot['x'], "top": spot['y'],
                "width": spot['w'], "height": spot['h'],
                "stroke": "#00FF00", "strokeWidth": 2, "fill": "rgba(0, 255, 0, 0.2)"
            })

        # Draw Canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=bg_image,
            update_streamlit=True,
            height=480, width=640,
            drawing_mode="rect",
            initial_drawing=initial_drawing,
            key="canvas",
        )

        # Save Logic
        if st.button("💾 Save Configuration"):
            if canvas_result.json_data is not None:
                new_spots = []
                objects = canvas_result.json_data["objects"]
                for i, obj in enumerate(objects):
                    new_spots.append({
                        "id": f"A{i+1}", # Auto-generate IDs A1, A2...
                        "x": int(obj["left"]),
                        "y": int(obj["top"]),
                        "w": int(obj["width"]),
                        "height": int(obj["height"]) # Note: canvas uses 'height', we use 'h'
                    })
                
                # Update System Config
                new_config = system.config
                new_config["spots"] = new_spots
                system.save_config(new_config)
                st.success(f"Saved {len(new_spots)} parking spots!")

else:
    # LIVE MONITOR MODE
    st.header("Live Feed & Detection")
    st.write("Monitoring real-time occupancy...")
    
    image_placeholder = st.empty()
    
    while True:
        frame = system.get_frame()
        if frame is not None:
            image_placeholder.image(frame, channels="RGB")
        time.sleep(0.05)