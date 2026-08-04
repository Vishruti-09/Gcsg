# Gesture Control Snake Game

A computer vision-based Snake Game that allows players to control the game using **hand gestures instead of a physical keyboard or controller**. The system uses a webcam to capture hand gestures, processes the captured image using OpenCV, and uses a trained VGG-based deep learning model to recognize gestures and convert them into directional commands for controlling the Snake.

## Overview

Traditional computer games generally require physical input devices such as keyboards, mice, or gaming controllers. This project demonstrates an alternative human-computer interaction approach in which a player can control a computer game using **hand gestures**.

The system is inspired by technologies such as **Leap Motion**, which allow computers to recognize and interpret human hand movements. In this project, a webcam is used to capture the player's hand, and computer vision techniques are applied to isolate and preprocess the hand region. A trained deep learning model then classifies the gesture and maps it to a corresponding Snake movement.

The application combines **computer vision, deep learning, and game development** to create a controller-free gaming experience.

## Features

* Gesture-based Snake game control
* Webcam-based hand gesture recognition
* Real-time video capture using OpenCV
* Background subtraction for hand isolation
* Image preprocessing using grayscale conversion, Gaussian blur, and thresholding
* Contour and convex hull detection
* VGG-based trained deep learning model for gesture classification
* Five supported hand gestures
* Automatic mapping of gestures to keyboard/game directions
* Snake movement and collision detection
* Controller-free gameplay

## Tech Stack

| Category         | Technologies                         |
| ---------------- | ------------------------------------ |
| Language         | Python                               |
| Deep Learning    | Keras, VGG-based CNN Model           |
| Computer Vision  | OpenCV                               |
| Image Processing | NumPy, imutils                       |
| Game Development | Pygame                               |
| Machine Learning | Trained Gesture Classification Model |
| Camera Input     | Webcam                               |
| Randomization    | Python Random Module                 |


## Project Structure

```text
Gesture-Control-Snake-Game/
│
├── src/
│   └── main.py
│
├── models/
│   └── VGG_cross_validated.h5
│
├── dataset/
│   └── Gesture Images/
│
├── assets/
│   └── Game Resources/
│
├── screenshots/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```


---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Gesture-Control-Snake-Game
```

### Install Dependencies

Install the required Python libraries:

```bash
pip install opencv-python
pip install numpy
pip install keras
pip install tensorflow
pip install pygame
pip install imutils
```

## Running the Application

Start the application using:

```bash
python main.py
```

Make sure that:

* A working webcam is connected.
* The trained model file `VGG_cross_validated.h5` is available in the expected location.
* The required Python libraries are installed.

---

## How the System Works

The system uses a webcam to capture the player's hand and follows a sequence of computer vision and machine learning operations to control the Snake.

### Workflow

```text
        Webcam
           │
           ▼
   Capture Video Frame
           │
           ▼
    Frame Preprocessing
           │
           ▼
   Background Subtraction
           │
           ▼
   Grayscale Conversion
           │
           ▼
     Gaussian Blur
           │
           ▼
       Thresholding
           │
           ▼
    Contour Detection
           │
           ▼
    Hand Region Extraction
           │
           ▼
    Image Resizing (224×224)
           │
           ▼
    VGG-Based ML Model
           │
           ▼
    Gesture Classification
           │
           ▼
     Gesture → Direction
           │
           ▼
       Snake Movement
           │
           ▼
      Game Interaction
```

The implementation captures frames from the webcam, flips and smooths the image, and extracts a region of interest for gesture processing. When the background is captured, a background-subtraction model is applied before thresholding and contour detection.

---

## Gesture Recognition

The system recognizes five hand gestures:

| Gesture | Game Action   |
| ------- | ------------- |
| Fist    | Exit / Escape |
| L       | Move Up       |
| Okay    | Move Down     |
| Palm    | Move Right    |
| Peace   | Move Left     |

The gesture classes are defined in the application as:

```python
gesture_names = {
    0: 'Fist',
    1: 'L',
    2: 'Okay',
    3: 'Palm',
    4: 'Peace'
}
```

The recognized gesture is subsequently decoded into a corresponding game direction.

---

## Gesture Detection Pipeline

### 1. Webcam Capture

The application initializes the computer's camera using OpenCV:

```python
camera = cv2.VideoCapture(0)
```

The camera continuously captures frames while the application is running.

### 2. Frame Preprocessing

The captured frame is resized, smoothed using a bilateral filter, and horizontally flipped to provide a more natural interaction experience.

```python
frame = imutils.resize(frame, width=700)
frame = cv2.bilateralFilter(frame, 5, 50, 100)
frame = cv2.flip(frame, 1)
```

### 3. Background Removal

The system uses OpenCV's MOG2 background-subtraction method to isolate the hand from the background.

```python
bgModel = cv2.createBackgroundSubtractorMOG2(
    0,
    bgSubThreshold
)
```

The resulting foreground mask is then processed to reduce noise.

### 4. Image Thresholding

The extracted hand region is converted to grayscale and blurred before binary thresholding is applied.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(
    gray,
    (blurValue, blurValue),
    0
)

ret, thresh = cv2.threshold(
    blur,
    threshold,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

### 5. Contour Detection

The largest contour is identified as the primary hand region. A convex hull is also calculated around the detected contour.

```python
contours, hierarchy = cv2.findContours(
    thresh1,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

hull = cv2.convexHull(res)
```

This allows the system to visualize and process the detected hand shape.

---

## Deep Learning Model

The gesture recognition component uses a trained **VGG-based model** stored as:

```text
VGG_cross_validated.h5
```

The model is loaded using Keras:

```python
model = load_model('VGG_cross_validated.h5')
```

Before prediction, the processed gesture image is:

1. Converted into a NumPy array
2. Normalized to a range of 0–1
3. Resized to `224 × 224`
4. Converted into a three-channel image
5. Passed to the trained model

```python
target = np.stack((thresh,) * 3, axis=-1)
target = cv2.resize(target, (224, 224))
target = target.reshape(1, 224, 224, 3)

prediction, score = predict_rgb_image_vgg(target)
```

The predicted class is determined using the class with the highest model output probability.

---

## Snake Game

The game is developed using **Pygame**.

The game screen is initialized with a resolution of:

```text
800 × 600 pixels
```

The main game components include:

* Snake
* Snake segments
* Apple/food
* Score
* Game timer
* Collision detection
* Boundary handling
* Game-over functionality

### Snake Movement

The snake supports four primary directions:

```text
UP
DOWN
LEFT
RIGHT
```

The `Snake` class manages the snake's position, direction, body segments, movement, growth, and collision detection.

The system also prevents the snake from making an immediate 180-degree turn.

## Controls

The primary control mechanism is **hand gesture recognition**.

| Gesture    | Action |
| ---------- | ------ |
| L          | Up     |
| Okay       | Down   |
| Palm       | Right  |
| Peace      | Left   |
| Fist / ESC | Exit   |

The application also uses keyboard input during the setup and camera-processing stages, including:

| Key     | Function                            |
| ------- | ----------------------------------- |
| `B`     | Capture background                  |
| `SPACE` | Process/predict the current gesture |
| `ESC`   | Exit                                |

The gesture-to-direction mapping is implemented through the `decode_gesture_names` dictionary and the `getKey()` function.

---

## Game Interface

The Pygame interface provides:

* Game window
* Snake display
* Current score

## System Architecture

```text
┌──────────────────────┐
│      Webcam          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Frame Acquisition    │
│ & Preprocessing      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Background           │
│ Subtraction          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Hand Region          │
│ Extraction           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Thresholding &       │
│ Contour Detection    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ VGG-Based Gesture    │
│ Classification       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Gesture-to-Direction │
│ Mapping              │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Snake Game       │
│      (Pygame)        │
└──────────────────────┘
```

---

## Key Algorithms

### Background Subtraction

Used to separate the foreground hand from the background.

### Gaussian Blur

Used to smooth the extracted image and reduce noise before thresholding.

### Binary Thresholding

Converts the processed grayscale image into a binary representation suitable for hand-shape detection.

### Contour Detection

Identifies the boundaries of the detected hand region.

### Convex Hull

Creates an enclosing boundary around the detected hand contour.

### VGG-Based Classification

Classifies the processed hand image into one of the five predefined gesture categories.

### Collision Detection

Determines whether the Snake's head has collided with an apple or one of its body segments.

---

## Project Objectives

### Primary Objective

To develop a computer game that can be controlled using **human hand gestures** instead of a conventional physical controller.

### Secondary Objective

To demonstrate how computer vision and machine learning can be used to create a **controller-free human-computer interaction system**.

### Overall Goal

The main goal of the project is to create a gesture-controlled gaming system inspired by technologies such as Leap Motion, where human hand gestures can be detected and interpreted as computer commands.

---

## Advantages

* Eliminates the need for a physical game controller
* Provides a natural form of human-computer interaction
* Demonstrates practical computer vision applications
* Combines deep learning with game development
* Can be operated using a standard webcam
* Provides an interactive demonstration of gesture recognition

---

## Applications

The concepts demonstrated in this project can be extended to:

* Gesture-controlled games
* Human-computer interaction systems
* Virtual controllers
* Smart interfaces
* Touchless computer interaction
* Educational computer vision applications
* Interactive entertainment systems

---

## Limitations

The current implementation depends on:

* A functioning webcam
* Appropriate lighting conditions
* Proper background capture
* Correct positioning of the hand within the region of interest
* Availability of the trained gesture recognition model

The gesture classification is also based on a predefined set of five gestures rather than a general-purpose hand gesture recognition system. These characteristics follow the current implementation.

---

## Future Improvements

* Real-time continuous gesture recognition without requiring manual prediction triggering
* Improved hand detection using modern hand-tracking frameworks
* Integration of MediaPipe Hands
* YOLO-based hand/object detection
* Addition of more gestures and game controls
* Improved gesture classification accuracy
* Elimination of manual background capture
* Real-time gesture confidence visualization
* Support for multiple games using the same gesture controller
* Improved collision and game-over handling
* Full-screen gaming support
* Deployment as a standalone desktop application
* Optimization for low-end devices
* Integration with modern deep learning architectures
* Support for custom user-defined gestures

---

## Screenshots


<img width="1600" height="686" alt="GCSG 3" src="https://github.com/user-attachments/assets/82dd43ac-3a43-4db7-b107-0251d9e506d3" />
<img width="1600" height="900" alt="GCSG 2" src="https://github.com/user-attachments/assets/9487b8f8-9d89-45d4-9bc4-fc73c8e30599" />
<img width="1024" height="576" alt="GCSG 1" src="https://github.com/user-attachments/assets/c4b44591-4e5b-49dd-ab61-2c8dd2741fe9" />
<img width="473" height="190" alt="prop" src="https://github.com/user-attachments/assets/716937ed-a69d-4293-b0a6-90d668dda8c1" />
<img width="644" height="446" alt="uml" src="https://github.com/user-attachments/assets/4303a5ee-7c5e-4a87-94c8-13956e330c83" />




## Requirements

The project requires Python and the following major libraries:

```text
Python
OpenCV
NumPy
Keras
TensorFlow
Pygame
imutils
```

The application also requires the trained model:

```text
VGG_cross_validated.h5
```

---

## Conclusion

The **Gesture Control Snake Game** demonstrates how computer vision and deep learning can be combined with traditional game development to create a controller-free gaming experience.

The system captures hand gestures through a webcam, processes the hand region using OpenCV, classifies the gesture using a trained VGG-based model, and translates the recognized gesture into a directional command for the Snake game.

The project provides a practical demonstration of **gesture recognition, image processing, deep learning, and human-computer interaction** in a single interactive application.

---

## License

This project was developed as part of a Bachelor's Pre-Final Year Major Project and is intended for educational and research purposes.
