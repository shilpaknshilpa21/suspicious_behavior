import os, sys
import cv2     # <-- You missed this line

# Add src folder to Python path correctly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


from vision.detector import load_person_detector, detect_persons
from vision.pose_action import PoseActionDetector
from nlp.intent_reasoner import decide_intent
from nlp.alert_gen import generate_alert
from utils.drawer import draw_boxes, put_status


def analyze_video(video_path):
    if not os.path.exists(video_path):
        print("Video missing!", video_path)
        return
    
    cap = cv2.VideoCapture(video_path)
    hog = load_person_detector()
    pose_det = PoseActionDetector()
    
    actions = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        boxes = detect_persons(frame, hog)
        action = pose_det.classify_frame_action(frame)
        if action != "no_person":
            actions.append(action)

        draw_boxes(frame, boxes)
        put_status(frame, f"Action: {action}")

        cv2.imshow("Behavior Analysis", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    label, info = decide_intent(actions)
    alert = generate_alert(label, info)

    print("\n=== FINAL DECISION ===")
    print(alert)


if __name__ == "__main__":
    video_path = os.path.join("dataset", "normal", "Normal (2).mp4")
    analyze_video(video_path)
