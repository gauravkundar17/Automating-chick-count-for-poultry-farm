# 🐥 Automating Chick Counter for Poultry Farms (Comprehensive Guide)

An end-to-end computer vision and hardware integration solution for automating chick counting in poultry farming environments using **YOLOv8**, **OpenCV**, **Python**, and **Serial Communication (Arduino/Microcontrollers)**.

---

## 📋 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Detailed System Architecture](#-detailed-system-architecture)
3. [Deep-Dive Component Breakdown](#-deep-dive-component-breakdown)
   - [1. Dataset & Frame Extraction (`extract_frames.py`)](#1-dataset--frame-extraction-extract_framespy)
   - [2. YOLOv8 Training (`train.py`)](#2-yolov8-training-trainpy)
   - [3. Object Tracking & Counting (`custom_chick_counter.py`)](#3-object-tracking--counting-custom_chick_counterpy)
   - [4. Web Dashboard Interface (`templates/index.html`)](#4-web-dashboard-interface-templatesindexhtml)
4. [How Unique Counting Works (Avoiding Double Counts)](#-how-unique-counting-works-avoiding-double-counts)
5. [Hardware Integration Guide (Arduino / ESP32)](#-hardware-integration-guide-arduino--esp32)
6. [Step-by-Step Installation & Setup](#-step-by-step-installation--setup)
7. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
8. [Configuration & Performance Tuning](#-configuration--performance-tuning)
9. [Troubleshooting Guide](#-troubleshooting-guide)

---

## 📌 Executive Summary

Manual chick counting in commercial hatcheries and poultry farms is labor-intensive, error-prone, and stressful for young chicks. 

This project delivers an automated solution that:
- Captures overhead video feeds of chicks moving along conveyor belts, chutes, or farm pens.
- Uses **YOLOv8** deep learning object detection fine-tuned on custom chick image datasets.
- Implements **Multi-Object Tracking (MOT)** to track individual chicks frame-by-frame and assign unique IDs.
- Calculates an exact cumulative count of unique chicks without double-counting.
- Transmits real-time counts via **Serial Communication (UART / USB)** to hardware peripherals (e.g., LED matrix displays, Arduino microcontrollers, automated counting gates).

---

## 🏗️ Detailed System Architecture

```
                                 PHYSICAL LAYER
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  🎥 Overhead Camera / Video Stream ("videos/chicks.mp4" or USB Webcam)  │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │ Raw Video Frames (900x600)
                                       ▼
                             COMPUTER VISION LAYER
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  1. Frame Preprocessing (OpenCV)                                         │
  │     └─ Resize frame, normalize dimensions                              │
  │  2. YOLOv8 Object Detection & Tracking (Ultralytics)                     │
  │     ├─ Model: Fine-tuned weights ("runs/detect/chick_detector/weights/best.pt")│
  │     └─ Tracker: Persistent Multi-Object Tracking (Track ID assignment)   │
  │  3. Spatial-Temporal Unique Counter                                      │
  │     ├─ Extracts track IDs & confidence scores (> 0.5 filter)             │
  │     ├─ Compares track ID against `counted_ids = set()`                   │
  │     └─ Increments cumulative total on newly encountered IDs              │
  │  4. Frame Annotation & Visualization                                     │
  │     └─ Bounding box overlay, track ID label, total count banner          │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │ Serial Data: "count\n" (115200 Baud)
                                       ▼
                                HARDWARE LAYER
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  🔌 Serial Interface (COM3 / /dev/ttyUSB0)                               │
  │     ▼                                                                    │
  │  📟 Hardware Microcontroller (Arduino / ESP32 / Raspberry Pi)            │
  │     ├─ Drives LED Display / 7-Segment Counter                             │
  │     └─ Controls sorting gate / relay mechanism                           │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Deep-Dive Component Breakdown

### 1. Dataset & Frame Extraction (`extract_frames.py`)

When training custom computer vision models, collecting representative image frames from video footage is necessary. `extract_frames.py` automates this process:

- **Source Video**: `videos/chicks.mp4`
- **Output Folder**: `dataset/images/`
- **Sampling Logic**: Saves every **10th frame** (`count % 10 == 0`) to prevent near-identical duplicate frames in the dataset.

#### Code Explanation:
```python
import cv2
import os

video_path = "videos/chicks.mp4"
output_folder = "dataset/images"
os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)

count = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    if count % 10 == 0:  # Select frame interval
        filename = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(filename, frame)
    count += 1

cap.release()
```

---

### 2. YOLOv8 Training (`train.py`)

`train.py` fine-tunes a pretrained YOLOv8 Nano architecture (`yolov8n.pt`) on the annotated dataset configured in `dataset/data.yaml`.

#### Model Configuration (`dataset/data.yaml`):
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1           # Number of classes
names: ['chick'] # Class names
```

#### Training Script (`train.py`):
```python
from ultralytics import YOLO

# 1. Load pretrained YOLOv8 Nano model weights
model = YOLO("yolov8n.pt")

# 2. Start training pipeline
model.train(
    data="dataset/data.yaml", # Path to dataset config
    epochs=50,                # Number of training epochs
    imgsz=640,                # Input image dimension resolution
    batch=8,                  # Batch size per iteration
    name="chick_detector"     # Experiment output run folder name
)
```

**Training Outputs**:
- Saved run directory: `runs/detect/chick_detector/`
- Best model checkpoint: `runs/detect/chick_detector/weights/best.pt`
- Training metrics: Precision-Recall curves, confusion matrices, and loss charts (`results.csv`, `results.png`).

---

### 3. Object Tracking & Counting (`custom_chick_counter.py`)

This is the primary runtime application. It connects computer vision detection with real-time multi-object tracking and serial transmission.

#### Step-by-Step Line Breakdown:

1. **Imports & Initialization**:
   ```python
   from ultralytics import YOLO
   import cv2
   import serial
   import time

   # Load custom trained model
   model = YOLO("runs/detect/chick_detector/weights/best.pt")

   # Initialize Serial Port (COM3 at 115200 Baud)
   ser = serial.Serial('COM3', 115200)
   time.sleep(2)  # 2-second sleep allows Arduino serial auto-reset to settle

   # Open Video Stream
   cap = cv2.VideoCapture("videos/chicks.mp4")

   # Unique track ID repository
   counted_ids = set()
   ```

2. **Frame Processing Loop**:
   ```python
   while True:
       ret, frame = cap.read()
       if not ret:
           break

       # Resize frame for uniform display and processing speed
       frame = cv2.resize(frame, (900, 600))

       # Run YOLO Tracking with persistent ID retention across frames
       results = model.track(
           frame,
           persist=True,  # Maintains object identity across frames
           conf=0.8,      # Minimum confidence threshold for tracker creation
           classes=[0]    # Class 0: 'chick'
       )
   ```

3. **Detection Extraction & Counting Logic**:
   ```python
       if results[0].boxes.id is not None:
           boxes = results[0].boxes.xyxy.cpu().numpy()
           ids = results[0].boxes.id.cpu().numpy()
           confs = results[0].boxes.conf.cpu().numpy()

           for box, track_id, confidence in zip(boxes, ids, confs):
               if confidence < 0.5:
                   continue

               x1, y1, x2, y2 = map(int, box)
               track_id = int(track_id)

               # Check if this tracking ID has been counted before
               if track_id not in counted_ids:
                   counted_ids.add(track_id)
                   total_count = len(counted_ids)

                   # Write updated count to Serial interface
                   ser.write(f"{total_count}\n".encode())
                   print("Total Chicks:", total_count)

               # Draw bounding box around detected chick
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   ```

4. **HUD & Display**:
   ```python
       # Display total cumulative count on screen
       total_count = len(counted_ids)
       cv2.putText(frame, f"Total Chicks: {total_count}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

       cv2.imshow("Automated Chick Counter", frame)

       # Press ESC key (27) to exit loop
       if cv2.waitKey(1) == 27:
           break

   # Cleanup resources
   cap.release()
   cv2.destroyAllWindows()
   ser.close()
   ```

---

### 4. Web Dashboard Interface (`templates/index.html`)

The repository includes a web dashboard UI template (`templates/index.html`) designed for monitoring the system remotely via a web browser (e.g. integrated with a Flask or FastAPI backend):

- **Status Indicator**: Displays current system state (`Running`, `Stopped`).
- **Live Counter Banner**: Shows large formatted numerical count.
- **Control Buttons**:
  - `▶ Start Counting`: Triggers process initialization (`/start`).
  - `⏹ Stop Counting`: Pauses processing (`/stop`).
  - `🔄 Reset Count`: Clears accumulated count set (`/reset`).
- **Video Feed Embed**: Standard HTML image stream endpoint (`<img src="/video">`).

---

## 🎯 How Unique Counting Works (Avoiding Double Counts)

A major challenge in object counting in video feeds is preventing double counting when objects stay in the camera view for multiple frames or briefly re-enter.

### The Problem without Tracking:
If you simply count detected bounding boxes in each frame (`len(boxes)`), a single chick visible for 100 frames would be counted **100 times**.

### The Solution: Multi-Object Tracking + Persistent Set Data Structure
1. **YOLOv8 Tracking (`persist=True`)**:
   - YOLOv8 integrates algorithms like **BoT-SORT** / **ByteTrack**.
   - As chicks move, the algorithm calculates spatial position, trajectory vectors, and feature embeddings.
   - It assigns a unique numeric **Tracking ID** (e.g., `ID 1`, `ID 2`, `ID 3`) to each individual chick and maintains that ID continuously across frames.

2. **Python `set()` Filtering**:
   - A Python `set` structure (`counted_ids = set()`) only stores unique values.
   - When `track_id = 5` is detected:
     - **Frame 1**: `5 not in counted_ids` $\rightarrow$ Added to set $\rightarrow$ `counted_ids = {5}` $\rightarrow$ **Count = 1** (Transmitted over Serial).
     - **Frame 2 to Frame 100**: `5 in counted_ids` $\rightarrow$ Condition `if track_id not in counted_ids` fails $\rightarrow$ **Count remains 1** (No duplicate counts or serial triggers).

---

## 🔌 Hardware Integration Guide (Arduino / ESP32)

To display counts on physical hardware or operate an automated sorting gate, connect a microcontroller via USB to your host computer running `custom_chick_counter.py`.

### 1. Hardware Setup Diagram

```
 [ Computer (Running custom_chick_counter.py) ]
                    │
               USB Cable (Serial @ COM3 / 115200 Baud)
                    │
                    ▼
          [ Arduino / ESP32 Board ]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
 [ 16x2 LCD / 7-Segment ]  [ Relay / Buzzer / Servo Gate ]
```

### 2. Sample Arduino C++ Code (`chick_counter_receiver.ino`)

Upload this code to your Arduino board using the Arduino IDE:

```cpp
// Chick Counter Receiver Code for Arduino
#include <LiquidCrystal.h>

// Initialize LCD (RS, E, D4, D5, D6, D7)
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

int count = 0;

void setup() {
  // Initialize Serial interface matching python baud rate (115200)
  Serial.begin(115200);
  
  // Initialize LCD display
  lcd.begin(16, 2);
  lcd.print("CHICK COUNTER");
  lcd.setCursor(0, 1);
  lcd.print("Count: 0");
}

void loop() {
  // Check if serial data is available from Python script
  if (Serial.available() > 0) {
    // Read serial string until newline character '\n'
    String data = Serial.readStringUntil('\n');
    count = data.toInt();

    // Update LCD display
    lcd.setCursor(0, 1);
    lcd.print("Count: ");
    lcd.print(count);
    lcd.print("       "); // Clear trailing characters
  }
}
```

---

## ⚙️ Step-by-Step Installation & Setup

### 1. Clone or Open Workspace
Open a terminal (PowerShell / Command Prompt) in the project directory:
```bash
cd c:\Users\hp\Desktop\chick_counter_project
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install ultralytics opencv-python pyserial torch torchvision pyyaml
```

*(Optional GPU Acceleration)*: If you have an NVIDIA GPU, install PyTorch with CUDA support for faster processing:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Step-by-Step Execution Guide

### Workflow Step 1: Prepare Raw Video & Extract Dataset Frames
Place your poultry video into `videos/chicks.mp4`, then run:
```bash
python extract_frames.py
```
*Output*: Sample frame images will be written to `dataset/images/`.

---

### Workflow Step 2: Model Training
Train the YOLOv8 detector using your annotated dataset:
```bash
python train.py
```
*Outputs*:
- Training progress is printed to terminal.
- Weights are saved at `runs/detect/chick_detector/weights/best.pt`.

---

### Workflow Step 3: Run Real-Time Counting & Hardware Output
Ensure your Arduino/ESP32 is plugged in and verify the COM port (e.g. `COM3` on Windows or `/dev/ttyUSB0` on Linux).

Run the counter script:
```bash
python custom_chick_counter.py
```
- A window titled **"Automated Chick Counter"** will display the live video feed with bounding boxes, unique IDs, and total count.
- Updated counts will be transmitted live over the serial interface and logged in terminal.
- Press <kbd>ESC</kbd> to safely close the application and serial port.

---

## ⚙️ Configuration & Performance Tuning

All key hyperparameters in `custom_chick_counter.py` can be customized for your farm setup:

| Variable | Current Setting | Purpose & How to Adjust |
| :--- | :--- | :--- |
| `ser = serial.Serial('COM3', 115200)` | Port `COM3`, Baud `115200` | Change `'COM3'` to match your Device Manager COM port (or `'/dev/ttyACM0'` on Linux). |
| `cap = cv2.VideoCapture(...)` | `"videos/chicks.mp4"` | Change to `0` or `1` to use a connected live USB/RTSP security camera feed instead of video file. |
| `cv2.resize(frame, (900, 600))` | `(900, 600)` | Adjust resolution. Lower values (e.g. `640, 480`) increase FPS on low-power hardware (Raspberry Pi). |
| `model.track(..., conf=0.8)` | `0.8` | Minimum detection confidence score to initiate tracking. Decrease (e.g. `0.6`) if small chicks are missed; increase if false positives occur. |
| `if confidence < 0.5:` | `0.5` | Secondary filtering threshold for box rendering and count registration. |

---

## ❓ Troubleshooting Guide

### 1. `serial.serialutil.SerialException: could not open port 'COM3'`
- **Cause**: Arduino is not plugged in, wrong COM port specified, or another application (like Arduino IDE Serial Monitor) is using `COM3`.
- **Solution**:
  1. Open Windows Device Manager $\rightarrow$ Ports (COM & LPT) to find your exact port number.
  2. Close the Arduino IDE Serial Monitor window.
  3. Update `'COM3'` in `custom_chick_counter.py` to your port (e.g., `'COM4'`).

### 2. OpenCV Window Closes Immediately
- **Cause**: `videos/chicks.mp4` path is incorrect or file is unreadable.
- **Solution**: Verify `videos/chicks.mp4` exists in the working directory.

### 3. Model file not found (`FileNotFoundError: runs/detect/chick_detector/weights/best.pt`)
- **Cause**: Model has not been trained yet.
- **Solution**: Either run `python train.py` first to generate `best.pt`, or point the model loader in `custom_chick_counter.py` to base weights (`model = YOLO("models/yolov8n.pt")`).

### 4. Duplicate Counts / Lost Track IDs
- **Cause**: Chicks moving too fast or overlapping heavily (occlusion).
- **Solution**: Mount camera directly overhead (90-degree bird's-eye view), ensure consistent lighting, and adjust confidence thresholds (`conf=0.7`).

---

## 🏷️ Dataset Attribution & License

- **Dataset Source**: [Roboflow Universe - Chick Counter](https://universe.roboflow.com/gautam-malhan/chick_counter-n27zl/dataset/1)
- **Dataset License**: CC BY 4.0
- **Project License**: MIT License
