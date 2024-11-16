import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
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
    cv2.putText(frame, "Press 'c' to capture and save the image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the frame with bounding boxes and instructions
    cv2.imshow('Capture Image', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Capture image on 'c' key press
    if key == ord('c'):
        if len(faces) > 0:
            # Ask the user for the name to save the image
            name = input("Enter the name to save the image: ")
            x, y, w, h = faces[0]  # Get the first detected face
            cropped_face = frame[y:y + h, x:x + w]  # Crop the face region
            filename = f"{name}.jpg"  # Create a filename using the user's input
            cv2.imwrite(filename, cropped_face)  # Save the cropped face
            print(f"Image captured and saved as '{filename}'.")
        else:
            print("No face detected. Please try again.")

    # Quit on 'q' key press
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
