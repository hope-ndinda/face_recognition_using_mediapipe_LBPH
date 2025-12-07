import cv2
import os
import mediapipe as mp

persons = ["Hope", "Jolie"]  # Your two people
output_dir = "dataset"
num_images = 200  # Images per person

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

for person in persons:
    print(f"\nReady to capture for: {person}")
    input("Press ENTER when the correct person is in front of the camera...")

    save_path = f"{output_dir}/{person}"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"\n📸 Capturing for {person} — Press 'q' to stop early.\n")

    while cap.isOpened() and count < num_images:
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

                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    file_path = f"{save_path}/{count}.jpg"
                    cv2.imwrite(file_path, face)
                    count += 1
                    cv2.putText(frame, f"{person} | {count}/{num_images}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow(f"Capturing {person}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✔ Completed capture for {person}!\n")

print("\n🎉 All image capture done successfully.")
