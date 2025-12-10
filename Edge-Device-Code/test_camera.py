import depthai as dai
import cv2
import time

print("1. Starting Camera Test...")

# Create pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(300, 300)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)

print("2. Pipeline created. Connecting to device...")

# Connect to device
try:
    with dai.Device(pipeline) as device:
        print("3. Device Connected! Starting Stream...")
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        # Run for 50 frames then stop
        for i in range(50):
            in_frame = q.get()  # Blocking call, waits for data
            print(f"   Received Frame {i+1}")
            
        print("4. Success! Camera is working.")
        
except Exception as e:
    print(f"ERROR: {e}")