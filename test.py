import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
import os

# Initialize FaceNet model
facenet = FaceNet()

# File path to save and load embeddings
EMBEDDINGS_FILE = "face_embeddings.pkl"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to store multiple embeddings for each person
reference_embeddings = {}

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Load embeddings from file if it exists
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

# Save embeddings to file
def save_embeddings():
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(reference_embeddings, f)
    print("Embeddings saved to file.")

# Load embeddings on startup
reference_embeddings = load_embeddings()
if reference_embeddings:
    print("Loaded saved embeddings. Ready for real-time face recognition.")
else:
    print("No saved embeddings found. Please capture reference images first.")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Step 1: Capture and Save Reference Images with Embeddings
print("Press 'c' to capture a reference image for a person within the bounding box and save it with their name.")
print("Press 'q' to quit capturing reference images and start real-time face recognition.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame.")
        break

    # Convert frame to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Draw bounding box around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display instructions on the frame
    cv2.putText(frame, "Press 'c' to capture reference image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to start recognition", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the frame with bounding boxes and instructions
    cv2.imshow('Capture Reference Image', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Capture image on 'c' key press
    if key == ord('c'):
        if len(faces) > 0:
            # Ask the user for the name to save the image under
            name = input("Enter the name for the reference image: ")

            # If the person is new, create an empty list for their embeddings
            if name not in reference_embeddings:
                reference_embeddings[name] = []

            # Extract the first detected face and get its embedding
            x, y, w, h = faces[0]
            cropped_face = frame[y:y + h, x:x + w]
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

            # Calculate the face embedding and store it
            face_embedding = facenet.extract(cropped_face_rgb, threshold=0.95)[0]['embedding']
            reference_embeddings[name].append(face_embedding)
            print(f"Reference image for {name} captured and embedding saved!")

            # Save embeddings after capturing
            save_embeddings()
        else:
            print("No face detected. Please try again.")

    # Start recognition on 'q' key press
    if key == ord('q'):
        print("Starting real-time face recognition.")
        break

cv2.destroyWindow('Capture Reference Image')

# Step 2: Real-Time Face Recognition
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Extract face region and compute embedding
        face_region = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_embeddings = facenet.extract(face_rgb, threshold=0.95)

        label = "Unknown"
        highest_similarity = 0.0

        # Compare detected face embedding to all stored embeddings for each person
        if face_embeddings:
            embedding = face_embeddings[0]['embedding']
            for name, embeddings in reference_embeddings.items():
                for ref_embedding in embeddings:
                    similarity = cosine_similarity(ref_embedding, embedding)
                    if similarity > 0.8 and similarity > highest_similarity:
                        label = name
                        highest_similarity = similarity

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {highest_similarity:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the resulting frame with recognition
    cv2.imshow('Real-Time Face Recognition', frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
