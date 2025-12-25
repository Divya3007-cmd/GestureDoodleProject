# gesture_doodle_record_improved.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ---------- Configuration ----------
OUTPUT_FILENAME = "output_recording.mp4"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 20.0
DRAW_COLOR = (0, 0, 255)     # Red
DRAW_THICKNESS = 6
ERASER_RADIUS_RATIO = 0.34
SMOOTHING_WINDOW = 3
DRAW_LANDMARKS = False
# -----------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_up(hand_landmarks, handedness_label=None):
    tips = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20}
    pips = {'thumb':2, 'index':6, 'middle':10, 'ring':14, 'pinky':18}
    res = {}
    lm = hand_landmarks.landmark

    for finger in ['index', 'middle', 'ring', 'pinky']:
        res[finger] = lm[tips[finger]].y < lm[pips[finger]].y

    if handedness_label:
        if handedness_label.lower().startswith('right'):
            res['thumb'] = lm[4].x < lm[2].x
        else:
            res['thumb'] = lm[4].x > lm[2].x
    else:
        res['thumb'] = abs(lm[4].x - lm[2].x) > 0.03

    return res

def hand_center_and_size(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    pts = [0, 1, 5, 9, 13, 17]
    cx = int(np.mean([lm[i].x for i in pts]) * w)
    cy = int(np.mean([lm[i].y for i in pts]) * h)

    wrist = np.array([lm[0].x, lm[0].y])
    mid = np.array([lm[9].x, lm[9].y])
    size = int(np.linalg.norm(wrist - mid) * np.sqrt(w*h))
    return (cx, cy), max(size, 20)

def circle_erase(canvas, center, radius):
    cv2.circle(canvas, center, radius, (0, 0, 0), -1)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_FILENAME,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (w, h)
    )

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    points = deque(maxlen=SMOOTHING_WINDOW)
    prev_point = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mode = "Idle"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                handed = results.multi_handedness[0].classification[0].label
                fu = fingers_up(hand, handed)
                count = sum(fu.values())

                lm = hand.landmark
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                (pcx, pcy), size = hand_center_and_size(hand, w, h)

                if fu['index'] and not fu['middle'] and not fu['ring'] and not fu['pinky']:
                    mode = "Draw"
                    points.append((ix, iy))
                    smooth = np.mean(points, axis=0).astype(int)

                    if prev_point is None:
                        prev_point = tuple(smooth)

                    cv2.line(canvas, prev_point, tuple(smooth),
                             DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)
                    prev_point = tuple(smooth)

                elif count >= 4:
                    mode = "Erase"
                    prev_point = None
                    points.clear()
                    radius = int(size * ERASER_RADIUS_RATIO)
                    circle_erase(canvas, (pcx, pcy), radius)
                    cv2.circle(frame, (pcx, pcy), radius, (0,255,255), 2)

                else:
                    prev_point = None
                    points.clear()

                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(frame, hand,
                                              mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Mode: {mode}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            final = cv2.addWeighted(frame, 0.7, canvas, 1.0, 0)
            cv2.imshow("Gesture Doodle Recorder", final)
            out.write(final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
