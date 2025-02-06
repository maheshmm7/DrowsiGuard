# DrowsiGaurd (Drowsiness Detection System)

This project implements a **Drowsiness Detection System** using **Computer Vision** and **Deep Learning** techniques to monitor and detect signs of drowsiness in drivers. The system tracks the driver's eye state and gaze direction using the webcam, and it raises an alert if drowsiness is detected, ensuring safer driving conditions.

## Features

- **Real-time Drowsiness Detection:** Tracks the eye state to detect when the driver is drowsy or blinking excessively.
- **Gaze Direction Tracking:** Monitors the direction of the gaze to determine if the driver is distracted.
- **Head Pose Estimation:** Analyzes the orientation of the driver's head to detect abnormal head movements or drowsiness.
- **Alert System:** Plays an alarm sound when drowsiness or distraction is detected.
- **Face and Eye Detection:** Utilizes Haar cascades for detecting faces and eyes.


## Prerequisites
- **Python Version**: Python 3.7 - 3.9 (Python < 3.10 required)
- Webcam for real-time monitoring
- The following dependencies, listed in requirements.txt file

## Setup and Installation

### 1. Using Virtual Environment (venv)

#### 1.1 Create Virtual Environment
```bash
python -m venv drowsiness_env
```

#### 1.2 Activate Virtual Environment
- **Windows**: 
  ```bash
  drowsiness_env\Scripts\activate
  ```
- **macOS/Linux**: 
  ```bash
  source drowsiness_env/bin/activate
  ```

#### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
pip install protobuf==3.20.0
```

### 2. Using Conda Environment

#### 2.1 Create Conda Environment
```bash
conda create --name drowsiness_env python=3.9 -y
```

#### 2.2 Activate Conda Environment
```bash
conda activate drowsiness_env
```

#### 2.3 Install Dependencies
```bash
pip install -r requirements.txt
pip install protobuf==3.20.0
```

### 3. Verify Setup

Check installed packages:
```bash
pip list
```

## Recommendations

- **Conda is recommended** for managing complex dependencies, especially for machine learning projects with potential GPU support.
- Always ensure you're in the correct virtual environment before running the project.
- Keep your `requirements.txt` updated with all necessary dependencies.

## Troubleshooting

- If you encounter any installation issues, ensure you have the latest version of pip:
  ```bash
  pip install --upgrade pip
  ```
- Check Python version compatibility (Python 3.7 - 3.9 recommended)
- Verify all system dependencies are installed

Required libraries include:
```text
opencv-python
mediapipe
numpy
keras
pygame
```

### 4. Model and Haar Cascade Files
Ensure the following files are available in the project directory:

Pre-trained model for eye state classification:
- `cnnfinal.h5`

Haar cascade files for face and eye detection:
- `haarcascade_frontalface_alt.xml`
- `haarcascade_lefteye_2splits.xml`
- `haarcascade_righteye_2splits.xml`

These files can be downloaded from the official OpenCV repository or extracted from your local OpenCV installation.

### 5. Alarm Sound
Place an `alarm.wav` file in the project directory. This sound will be played when the system detects drowsiness or distraction.

## Usage

### 1. Running the Script
To start the drowsiness detection system, run:
```bash
python test.py
```
This will open the webcam feed and start tracking the driver's eye state, gaze direction, and head pose in real-time.

### 2. How It Works
- **Eye State Detection**: The system uses a Convolutional Neural Network (CNN) to classify whether the driver's eyes are open or closed.
- **Face and Eye Detection**: Haar cascades are used to detect the driver's face and eyes. The system tracks the eye region to monitor blinking and drowsiness.
- **Gaze Direction Tracking**: Using MediaPipe, the system estimates the gaze direction to detect whether the driver is distracted or looking away from the road.
- **Head Pose Estimation**: The system tracks the head's orientation to determine if the driver is nodding or tilting their head abnormally.
- **Alert System**: If drowsiness is detected (closed eyes for a certain duration or excessive blinking), an alarm sound is played.

### 3. Exiting the Program
To exit the program, press `q` in the terminal or stop the script execution.



## Troubleshooting

- **Webcam not detected**: Ensure that your webcam is properly connected and accessible by other applications. Try restarting your machine if the issue persists.
- **Model loading errors**: If you get an error related to loading the model (`cnnfinal.h5`), make sure the file is placed in the correct directory and the path is specified correctly in the script.

## Reporting Issues and Contributing

### Reporting Errors or Bugs

1. **Check Existing Issues**
   - Before reporting, search existing GitHub issues to avoid duplicates.

2. **Create a Detailed Bug Report**
   - Use GitHub Issues
   - Include:
     - Detailed description of the error
     - Steps to reproduce
     - Your environment details (OS, Python version, etc.)
     - Full error traceback


- Discuss in project discussions/comments

## Contributing

We welcome contributions to this project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes.
4. Push your changes and create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **OpenCV**: For providing the face and eye detection algorithms.
- **MediaPipe**: For real-time gaze and head pose tracking.
- **Keras**: For providing the deep learning framework used to train the eye state classification model.
