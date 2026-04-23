import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import json
import torch.nn as nn
from torchvision import models

# -----------------------------
# Page Config (Dark UI)
# -----------------------------
st.set_page_config(page_title="WildVision", layout="wide")

# -----------------------------
# Custom Dark Styling
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.title("🐾 WildVision - Real-Time Animal Detection")

# -----------------------------
# Load Models
# -----------------------------
yolo_model = YOLO("yolov8n.pt")

with open("../saved_models/classes.json", "r") as f:
    class_names = json.load(f)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("saved_models/animal_classifier.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

animal_classes = [14,15,16,17,18,19,20,21,22,23]

# -----------------------------
# Mode Selection
# -----------------------------
mode = st.radio("Choose Mode", ["Upload Image", "Use Webcam"])

# -----------------------------
# Detection Function
# -----------------------------
def detect(frame):
    results = yolo_model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if cls_id not in animal_classes or confidence < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (224, 224))
        img = Image.fromarray(crop)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf_resnet, predicted = torch.max(probs, 1)

        if conf_resnet.item() < 0.6:
            continue

        label = class_names[predicted.item()]
        yolo_label = yolo_model.names[cls_id]

        final_label = f"{yolo_label} → {label} ({conf_resnet.item():.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, final_label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame

# -----------------------------
# IMAGE UPLOAD MODE
# -----------------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = detect(frame)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        st.image(output, caption="Detection Result", use_column_width=True)
# -----------------------------
# WEBCAM MODE
# -----------------------------
if mode == "Use Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not working")
            break

        # Run detection
        output = detect(frame)

        # Convert BGR → RGB
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(output)

    cap.release()