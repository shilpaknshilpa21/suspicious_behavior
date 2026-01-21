import cv2

def load_person_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_persons(frame, hog):
    frame_small = cv2.resize(frame, (640, 360))
    rects, _ = hog.detectMultiScale(frame_small, winStride=(8, 8), padding=(8, 8), scale=1.05)

    sx = frame.shape[1] / 640
    sy = frame.shape[0] / 360

    boxes = []
    for (x, y, w, h) in rects:
        boxes.append((int(x * sx), int(y * sy), int(w * sx), int(h * sy)))
    return boxes
