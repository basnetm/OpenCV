import cv2
import numpy as np
import os  

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to training data folder
dataset_path = r"OpenCV\training_images"

# Empty lists to store faces and their corresponding labels
faces = []
labels = []

# Dictionary to map labels to person names (used for display)
label_to_name = {}

# Label counter starting from 0
current_label = 0

# Loop over each folder inside 'training_data'
for person_name in os.listdir(dataset_path):  
    person_folder = os.path.join(dataset_path, person_name)

    
    if not os.path.isdir(person_folder):
        continue

    label_to_name[current_label] = person_name

    # Loop over images inside that person's folder
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue  # Skip if image can't be loaded

        # Detect face in the image
        faces_rect = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_rect:
            # Crop and resize the face
            face_roi = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))

            # Store the face and its label
            faces.append(face_resized)
            labels.append(current_label)

    current_label += 1  

# Convert lists to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Train the recognizer
recognizer.train(faces, labels)
print("Training complete!")

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        # Crop and resize face
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))

        # Predict the label and confidence
        label, confidence = recognizer.predict(face_resized)

        # Map label to name
        person_name = label_to_name[label]

        # Color bounding box based on confidence
        if confidence >= 40:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green
            text = f"{person_name} ({round(confidence, 2)})"
            if person_name == "authorized_user_name":  # Replace with the authorized name
                print("Access Granted")
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red
            text = f"Unknown ({round(confidence, 2)})"

        # Display name and confidence on the frame
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

   
    cv2.imshow("Face Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
