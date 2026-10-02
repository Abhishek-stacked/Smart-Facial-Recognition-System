import os
import cv2
import numpy as np
from PIL import Image
import pickle
import time

# Start timer
start_time = time.time()

# Paths
BASE_DIR = "dataset"
TRAINED_MODEL_PATH = "trainer.yml"
LABELS_PATH = "labels.pickle"

# Create recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# ID Mapping
label_ids = {}
current_id = 0
x_train = []
y_labels = []
image_count = 0
face_found_count = 0
skipped_images = 0

print("[INFO] Starting face data processing...")

# Loop through folders and images
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            try:
                image = Image.open(path).convert("L")  # Convert to grayscale
                image = image.resize((550, 550))       # Resize for consistency
                image_array = np.array(image, "uint8")
                faces = detector.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)

                if len(faces) == 0:
                    skipped_images += 1

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    face_found_count += 1

            except Exception as e:
                print(f"[WARNING] Could not process image {path}: {e}")

            image_count += 1
            if image_count % 100 == 0:
                print(f"[INFO] Processed {image_count} images, {face_found_count} faces found...")

# Save label map
with open(LABELS_PATH, 'wb') as f:
    pickle.dump(label_ids, f)

# Train recognizer
print("[INFO] Training recognizer...")
recognizer.train(x_train, np.array(y_labels))
recognizer.save(TRAINED_MODEL_PATH)

end_time = time.time()
elapsed = round(end_time - start_time, 2)

# Summary
print("\n[INFO] Training Complete!")
print(f"Total Images Processed: {image_count}")
print(f"Faces Detected: {face_found_count}")
print(f"Images Skipped (no faces): {skipped_images}")
print(f"Unique People: {len(label_ids)}")
print(f"Model saved to: {TRAINED_MODEL_PATH}")
print(f"Labels saved to: {LABELS_PATH}")
print(f" Time Taken: {elapsed} seconds")
