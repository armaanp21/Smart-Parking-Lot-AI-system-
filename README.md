# Smart-Parking-Lot-AI-system-

Smart parking management system using YOLOv8, OAK-D / Raspberry Pi to detect free and occupied parking spots in real time. The system sends updates to a cloud-hosted API and allows a web/app interface to display current parking availability.

## Overview
This project uses a trained YOLOv8 model to identify parking spot occupancy from camera footage. The edge device (Raspberry Pi or OAK-D) runs inference and sends the results to the backend through API Gateway. A separate endpoint retrieves the latest parking data for use in apps or dashboards.


## Features
- Real-time detection of empty vs. occupied parking spots  
- YOLOv8 model trained on PKLot / Roboflow dataset  
- Edge device inference (Pi or OAK-D)  
- Backend built with AWS Lambda, DynamoDB, and API Gateway  
- Simple GET endpoint to fetch current parking status


Authors: Armaan, Naimul & Honor
