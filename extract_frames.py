import cv2
import os

video_path = "videos/chicks.mp4"
output_folder = "dataset/images"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Save every 10th frame
    if count % 10 == 0:
        filename = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(filename, frame)

    count += 1

cap.release()

print("Frames Extracted Successfully")