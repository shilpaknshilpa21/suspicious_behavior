import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseActionDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def classify_frame_action(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return "no_person"

        lm = results.pose_landmarks.landmark
        LW = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        LH = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y

        if LW > LH - 0.05:
            return "hand_near_pocket"

        return "normal"
