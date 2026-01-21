import cv2

def draw_boxes(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

def put_status(frame, text):
    y0 = 30
    for i, line in enumerate(text.split("\n")):
        y = y0 + i*20
        cv2.putText(frame, line, (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255), 2)
