# DrowsiGaurd (Drowsiness Detection System)

This project implements a **Drowsiness Detection System** using **Computer Vision** and **Deep Learning** techniques to monitor and detect signs of drowsiness in drivers. The system tracks the driver's eye state and gaze direction using the webcam, and it raises an alert if drowsiness is detected, ensuring safer driving conditions.

## Features

- **Real-time Drowsiness Detection:** Tracks the eye state to detect when the driver is drowsy or blinking excessively.
- **Gaze Direction Tracking:** Monitors the direction of the gaze to determine if the driver is distracted.
- **Head Pose Estimation:** Analyzes the orientation of the driver's head to detect abnormal head movements or drowsiness.
- **Alert System:** Plays an alarm sound when drowsiness or distraction is detected.
- **Face and Eye Detection:** Utilizes Haar cascades for detecting faces and eyes.

## Prerequisites

Before you begin, ensure that you have met the following requirements:
- Python 3.x (preferably 3.7 or higher)
- Webcam for real-time monitoring
- The following dependencies, listed in `requirements.txt` file

## Setup and Installation

### 1. Create a Virtual Environment
It is recommended to use a virtual environment for dependency management. You can create one using the following command:

```bash
python -m venv drowsiness_env
```

### 2. Activate the Virtual Environment
For Windows:
```bash
drowsiness_env\Scripts\activate
```

For macOS/Linux:
```bash
source drowsiness_env/bin/activate
```

### 3. Install Dependencies
Install the required dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt
```

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