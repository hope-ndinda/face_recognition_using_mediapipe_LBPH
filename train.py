import cv2
import os
import json
import numpy as np

dataset_dir = "dataset"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

labels = {}
faces = []
names = []

label_id = 0

for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path): continue

    labels[label_id] = person
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(img)
            names.append(label_id)

    label_id += 1

# Train LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(names))  # <--- fixed

recognizer.save(f"{model_dir}/lbph_model.xml")
with open(f"{model_dir}/label_map.json", "w") as f:
    json.dump(labels, f)

print("\nTraining Complete! Model saved in models/")