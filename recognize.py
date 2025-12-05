import cv2
import json
import mediapipe as mp

model = cv2.face.LBPHFaceRecognizer_create()
model.read("models/lbph_model.xml")

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb)

    if result.detections:
        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            h,w,_ = frame.shape
            x1,y1 = int(bbox.xmin*w), int(bbox.ymin*h)
            x2,y2 = x1+int(bbox.width*w), y1+int(bbox.height*h)

            face = frame[y1:y2,x1:x2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            label, confidence = model.predict(gray)
            label, confidence = model.predict(gray)

            if confidence < 70:     # lower = better match, adjust as needed
                name = label_map[str(label)]
            else:
                name = "Unknown"

            name = label_map[str(label)]

            cv2.putText(frame,f"{name} ({int(confidence)})",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break

cap.release()
cv2.destroyAllWindows()
