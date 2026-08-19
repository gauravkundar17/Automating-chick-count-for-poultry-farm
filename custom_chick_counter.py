from ultralytics import YOLO
import cv2
import serial
import time


model = YOLO("runs/detect/chick_detector/weights/best.pt")

ser = serial.Serial('COM3', 115200)

time.sleep(2)

cap = cv2.VideoCapture("videos/chicks.mp4")

counted_ids = set()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    
    frame = cv2.resize(frame, (900, 600))

    results = model.track(
        frame,
        persist=True,
        conf=0.8,
	classes=[0]
    )

    total_count = len(counted_ids)


    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        ids = results[0].boxes.id.cpu().numpy()

        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, confidence in zip(boxes, ids, confs):

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = box

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            track_id = int(track_id)

            if track_id not in counted_ids:

                counted_ids.add(track_id)

                total_count = len(counted_ids)

                ser.write(f"{total_count}\n".encode())

                print("Total Chicks:", total_count)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    total_count = len(counted_ids)

    cv2.putText(
        frame,
        f"Total Chicks: {total_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("Automated Chick Counter", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()

ser.close()