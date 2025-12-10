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



## Key Components
- **YOLOv8 Model** – Trained on the PKLot dataset (via Roboflow) to detect empty vs. occupied spaces.
- **Edge Device Pipeline**  
  - Captures frames from camera  
  - Runs local inference  
  - Sends updates to the cloud through an HTTP POST request  
- **Cloud Backend (AWS)**  
  - **Lambda (POST)** stores the latest parking status  
  - **DynamoDB** holds timestamped availability data  
  - **Lambda (GET)** returns the most recent parking availability for the UI  
  - **API Gateway** exposes both endpoints  
- **Frontend / App Layer**  
  - Displays vacancy information to users in real time  
  - Uses the GET endpoint to retrieve data

## How the System Works 
1. Camera or edge device captures a frame.  
2. YOLOv8 model predicts which spots are occupied.  
3. Results (spot counts, timestamp, etc.) are sent to AWS.  
4. Data is stored in DynamoDB.  
5. A client app calls the GET endpoint to show users the current availability.

## Technologies Used
- Python  
- YOLOv8 / Ultralytics  
- Roboflow (dataset preparation)  
- Raspberry Pi or OAK-D  
- AWS Lambda  
- AWS DynamoDB  
- AWS API Gateway
- 

## Objectives of the Project
- Demonstrate applied computer vision in a real-world scenario.  
- Show integration between edge computing and cloud services.  
- Build a scalable system capable of supporting real-time updates.  
- Provide a working prototype suitable for future expansion.



## Authors: Armaan, Naimul & Honor
