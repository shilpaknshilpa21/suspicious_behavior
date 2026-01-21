import os
import sys
import cv2
import uuid
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Add src path
BASE_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

from vision.detector import load_person_detector, detect_persons
from vision.pose_action import PoseActionDetector
from utils.drawer import draw_boxes, put_status
from nlp.intent_reasoner import decide_intent
from nlp.alert_gen import generate_alert


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


def analyze_and_save_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(3))
    height = int(cap.get(4))

    processed_name = f"processed_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, processed_name)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    hog = load_person_detector()
    pose_det = PoseActionDetector()

    actions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_persons(frame, hog)
        action = pose_det.classify_frame_action(frame)
        if action != "no_person":
            actions.append(action)

        draw_boxes(frame, boxes)
        put_status(frame, f"Action: {action}")

        out.write(frame)

    cap.release()
    out.release()

    label, info = decide_intent(actions)
    message = generate_alert(label, info)

    return output_path, label, message


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["video"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    output_video, label, message = analyze_and_save_video(save_path)
    video_processed_path = os.path.basename(output_video)

    return render_template(
        "index.html",
        video_name=filename,
        video_src=url_for("static", filename=f"processed/{video_processed_path}"),
        label=label,
        message=message
    )


if __name__ == "__main__":
    print("🚀 Running Flask server at: http://127.0.0.1:5000/")
    app.run(debug=True)
