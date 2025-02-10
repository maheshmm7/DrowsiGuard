# DrowsiGuard (Drowsiness Detection System)

> **Note:** This README is for the `logfeature` branch which includes additional terminal output for facial landmarks detection.

This project implements a **Drowsiness Detection System** using **Computer Vision** and **Deep Learning** techniques to monitor and detect signs of drowsiness in drivers. The system uses webcam input to track eye states, gaze direction, and facial landmarks, raising alerts when signs of drowsiness are detected.

## New Features in This Branch

- **Real-time Landmark Printing:** Continuously prints coordinates in terminal for:
  - Left eye landmarks
  - Right eye landmarks
  - Complete facial landmarks (x, y, z coordinates)
- All other features from the main branch remain unchanged

## Features

- **Real-time Eye State Monitoring:** Uses a CNN model to classify eyes as open or closed
- **Drowsiness Score System:** Maintains a drowsiness score based on eye state patterns
- **Gaze Direction Tracking:** Uses MediaPipe face mesh to monitor the driver's gaze direction
- **Alert System:** Plays an alarm sound when prolonged drowsiness is detected
- **Event Logging:** Records drowsiness events with timestamps
- **Image Capture:** Saves images when drowsy or sleepy states are detected
- **Visual Indicators:** Dynamic visual overlays indicating different drowsiness levels

## Prerequisites

- Python 3.7 - 3.9 (Python < 3.10 required)
- Webcam for real-time monitoring
- Required libraries (listed in requirements.txt)

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd drowsiguard
git checkout feature/landmark-printing
```

### 2. Using Virtual Environment (venv)

#### 2.1 Create Virtual Environment
```bash
python -m venv drowsiness_env
```

#### 2.2 Activate Virtual Environment
- **Windows**: 
  ```bash
  drowsiness_env\Scripts\activate
  ```
- **macOS/Linux**: 
  ```bash
  source drowsiness_env/bin/activate
  ```

#### 2.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install protobuf==3.20.0  # Required for MediaPipe compatibility
```

### 3. Using Conda Environment

#### 3.1 Create Conda Environment
```bash
conda create --name drowsiness_env python=3.9 -y
```

#### 3.2 Activate Conda Environment
```bash
conda activate drowsiness_env
```

#### 3.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install protobuf==3.20.0  # Required for MediaPipe compatibility
```

### 4. Verify Installation
```bash
pip list  # Check installed packages
```

### 5. Required Files

The system needs the following files in the project directory:

1. **Model File:**
   - `cnnfinal.h5` (Pre-trained CNN model for eye state classification)

2. **Haar Cascade Files:**
   - OpenCV's pre-trained Haar cascade files (automatically accessed via cv2.data.haarcascades)

3. **Audio File:**
   - `alarm.wav` (Sound file for drowsiness alerts)

## Usage

### Running the System

1. Start the program:
```bash
python drowdlog.py
```

2. The system will:
   - Access your webcam
   - Start monitoring eye states
   - Track gaze direction
   - Display real-time information on screen
   - Print landmark coordinates in the terminal:
     ```
     Face coordinates: x=224, y=167, w=232, h=232
     Right Eye coordinates: x=271, y=223, w=54, h=54
     1/1 [==============================] - 0s 25ms/step
     Left Eye coordinates: x=273, y=224, w=53, h=53
     1/1 [==============================] - 0s 32ms/step
     ```

## Terminal Output

When running the system, you'll see continuous output in the terminal showing:
- Face landmark coordinates (x, y, z)
- Left and right eye bounding box coordinates
- Detection status and changes

This feature is useful for:
- Debugging landmark detection
- Validating eye tracking accuracy
- Development and testing of new features

## Troubleshooting

Common issues and solutions:

1. **Webcam Not Detected:**
   - Check webcam connection
   - Verify webcam permissions
   - Try different video capture device indices

2. **Model Loading Errors:**
   - Ensure `cnnfinal.h5` is in the correct directory
   - Verify Keras version compatibility

3. **MediaPipe Errors:**
   - Check Python version compatibility
   - Verify MediaPipe installation

4. **Terminal Output Issues:**
   - If landmark coordinates aren't printing, check if MediaPipe face mesh initialization is successful
   - Verify console/terminal has sufficient scroll buffer for continuous output

## Contributing

Contributions are welcome! Please feel free to submit pull requests or report issues.

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenCV for computer vision capabilities
- MediaPipe for face mesh detection
- Keras for deep learning functionality
- pygame for audio alert system