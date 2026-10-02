# Face Detection Attendance System

<div align="center">

**A Python desktop attendance system that detects faces, recognizes registered users, and records attendance automatically in CSV format.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-LBPH%20Face%20Recognition-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Tkinter](https://img.shields.io/badge/GUI-Tkinter-FFB000?style=for-the-badge)](https://docs.python.org/3/library/tkinter.html)
[![CSV](https://img.shields.io/badge/Attendance-CSV-2EA44F?style=for-the-badge)](attendance.csv)

[Watch the demo video](test.mp4)

</div>

---

## Overview

This project is a face-recognition-based attendance system built with Python and OpenCV. It trains an LBPH face recognizer from a local image dataset, detects faces in real time through a webcam, identifies known users, and stores attendance records in `attendance.csv`.

The project includes both:

- A **desktop GUI** built with Tkinter: `gui_attendance.py`
- A **console/OpenCV window recognizer**: `recognize.py`

## Features

- Real-time webcam face detection
- LBPH face recognition using OpenCV
- Tkinter-based attendance interface
- Automatic attendance logging with date and time
- Duplicate attendance prevention
- Unknown face detection
- CSV-based attendance export
- Separate training and recognition scripts

## Project Structure

```text
.
+-- gui_attendance.py                  # Tkinter GUI attendance app
+-- recognize.py                       # Real-time recognition script
+-- train_model.py                     # Trains the face recognizer
+-- attendance.csv                     # Attendance output file
+-- test.mp4                           # Demo video
+-- Face_Detection_Attendance_System.pdf
+-- 01-134222-039-132601100893-17032025-114921pm.pdf
```

Generated after training:

```text
trainer.yml       # Trained LBPH model
labels.pickle     # Name-to-ID label mapping
```

Expected dataset layout:

```text
dataset/
+-- person_one/
|   +-- image1.jpg
|   +-- image2.jpg
|   +-- ...
+-- person_two/
|   +-- image1.jpg
|   +-- ...
+-- ...
```

Folder names become attendance labels. For example, `dataset/Ali Haider/` is stored as `ali_haider`.

## Tech Stack

| Tool | Purpose |
| --- | --- |
| Python | Core application language |
| OpenCV | Face detection and recognition |
| OpenCV Contrib | LBPH face recognizer |
| Haar Cascade | Frontal face detection |
| Tkinter | Desktop GUI |
| Pillow | Camera frame conversion for GUI display |
| NumPy | Image array processing |
| CSV | Attendance storage |

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/face-detection-attendance-system.git
cd face-detection-attendance-system
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install opencv-contrib-python pillow numpy
```

> `opencv-contrib-python` is required because the project uses `cv2.face.LBPHFaceRecognizer_create()`.

## How To Use

### 1. Prepare the dataset

Create a `dataset` folder and add one subfolder per person:

```text
dataset/
+-- your_name/
    +-- photo1.jpg
    +-- photo2.jpg
    +-- photo3.jpg
```

Use clear face images with different angles, lighting, and expressions for better recognition accuracy.

### 2. Train the model

```bash
python train_model.py
```

This creates:

- `trainer.yml`
- `labels.pickle`

### 3. Run the GUI app

```bash
python gui_attendance.py
```

In the app:

1. Click **Take Attendance** to start the webcam.
2. Wait for the system to detect and recognize a face.
3. Click **Register Attendance** to save the attendance record.
4. Click **See Attendance** to open `attendance.csv`.

### 4. Or run the console recognizer

```bash
python recognize.py
```

Press `q` to quit the recognition window.

## Attendance Output

Attendance is stored in `attendance.csv`:

```csv
Name,DateTime
ali_haider,2025-05-25 15:13:32
diyan_ehsan_syed,2025-05-26 22:56:30
```

## Demo

A test video is included with the project:

[Open demo video](test.mp4)

If GitHub does not preview the video directly, download or open `test.mp4` from the repository file list.

## Notes

- Make sure your webcam is connected and available.
- The app expects `trainer.yml` and `labels.pickle` to exist before running recognition.
- If recognition is inaccurate, add more training images and run `train_model.py` again.
- The recognition threshold is currently set to `conf < 70` in both recognition scripts.
- Attendance is marked once per name to avoid duplicate entries.

## Future Improvements

- Add a student/admin management screen
- Add daily attendance reset logic
- Export attendance by date range
- Add database support
- Add model confidence display in the GUI
- Package the app as a desktop executable

## Author

Developed as a face detection attendance system project using Python, OpenCV, and Tkinter.
