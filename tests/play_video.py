import cv2
import os

def play_video(path):
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    print(f"Playing: {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video Preview (ESC to exit)", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = os.path.join("dataset", "normal", "Normal (2).mp4")
    play_video(video_path)
