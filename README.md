This project is a real-time face recognition system using OpenCV and the LBPH algorithm. It detects faces from a webcam and identifies them based on trained images.

**Key Features:**

* Face detection using Haar Cascade
* Face recognition using LBPH
* Real-time webcam processing
* Displays name and confidence value
* Basic access control for authorized user

**Requirements:**

* Python
* OpenCV (`opencv-python` and `opencv-contrib-python`)
* NumPy

**How it works:**

* Loads images from dataset folder
* Trains the recognizer with labeled faces
* Detects faces from webcam
* Predicts identity and shows result

**Notes:**

* Lower confidence = better match
* Use clear and multiple images for accuracy
* Update authorized user name in code
