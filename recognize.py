import cv2
import pickle
import os
from datetime import datetime
import csv

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pickle")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascades",
    "haarcascade_frontalface_default.xml"
)
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance.csv")

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("[ERROR] Face cascade could not be loaded.")
    print(f"[ERROR] Expected file: {CASCADE_PATH}")
    exit()

# Load labels
with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

# Attendance file
attendance_file = ATTENDANCE_PATH

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "DateTime"])


def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Avoid duplicate marking
    with open(attendance_file, "r") as f:
        data = f.read()
        if name in data:
            return

    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now])
        print(f"[INFO] {name} marked present at {now}")


# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)

        if conf < 70:
            name = labels[id_]
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw box and name
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()

print("[INFO] Program ended.")
