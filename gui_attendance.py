import cv2
import pickle
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import csv
import os

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "DateTime"])

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("800x600")
root.configure(bg="#1c1b2f")

camera_label = tk.Label(root)
camera_label.pack(pady=20)

info_text = tk.Label(root, text="Click 'Take Attendance' to start camera and detect face", font=("Helvetica", 16), fg="white", bg="#1c1b2f")
info_text.pack()

# Globals
cap = None
detected_name = None
camera_running = False

def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_file, "r") as f:
        if name in f.read():
            return
    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now])
        messagebox.showinfo("Marked", f"{name} marked present at {now}")

def update_frame():
    global detected_name, camera_running, cap
    if not camera_running:
        return

    ret, frame = cap.read()
    if not ret:
        info_text.config(text="Failed to capture frame.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    detected_name = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)
        if conf < 70:
            name = labels[id_]
            detected_name = name
            info_text.config(text=f"Detected: {name}")
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            info_text.config(text="Unknown face")
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    root.after(15, update_frame)

def take_attendance_button_action():
    global cap, camera_running, detected_name

    if not camera_running:
        # Start camera and detection
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            return
        camera_running = True
        info_text.config(text="Camera started. Detecting face...")
        update_frame()
        take_btn.config(text="Register Attendance")
    else:
        # Camera running: try to mark attendance for detected face
        if detected_name:
            mark_attendance(detected_name)
        else:
            messagebox.showwarning("No Face", "No face recognized yet. Please wait or reposition yourself.")

def show_attendance():
    if os.path.exists(attendance_file):
        os.system(f"notepad {attendance_file}")
    else:
        messagebox.showinfo("Empty", "No attendance file found yet.")

def go_back():
    global cap, camera_running
    camera_running = False
    if cap:
        cap.release()
    root.destroy()

take_btn = tk.Button(root, text="Take Attendance", font=("Helvetica", 14), command=take_attendance_button_action, bg="#eeeeee", padx=10)
take_btn.pack(pady=10)

tk.Button(root, text="See Attendance", font=("Helvetica", 14), command=show_attendance, bg="#eeeeee", padx=10).pack(pady=5)
tk.Button(root, text="Go Back", font=("Helvetica", 14), command=go_back, bg="#eeeeee", padx=10).pack(pady=20)

root.mainloop()
