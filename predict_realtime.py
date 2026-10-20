import cv2
import mediapipe as mp
import pickle
import time
import pyttsx3
import math
import numpy as np
import os

# Load MLP model
with open("models/sign_model_mlp.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

text_buffer = ""
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

last_pred = None
start_time = 0

# Freeze gap control
freeze = False
freeze_start = 0
freeze_duration = 2  # 2 second gap after each character

box_x1, box_y1 = 150, 80
box_x2, box_y2 = 500, 400

def calc_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Box color feedback
    if freeze:
        box_color = (255, 0, 0)   # blue during freeze
    else:
        box_color = (0, 255, 0)   # green default
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)

    if results.multi_hand_landmarks and not freeze:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrist_px, wrist_py = int(wrist.x * w), int(wrist.y * h)

            if box_x1 < wrist_px < box_x2 and box_y1 < wrist_py < box_y2:
                middle_mcp = hand_landmarks.landmark[9]
                scale = calc_distance(wrist, middle_mcp) or 1e-6

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([(lm.x - wrist.x) / scale,
                                (lm.y - wrist.y) / scale,
                                (lm.z - wrist.z) / scale])

                # Predict with MLP and confidence
                proba = model.predict_proba([row])[0]
                confidence = np.max(proba)
                pred = model.classes_[proba.argmax()]
                current_time = time.time()

                if confidence > 0.7:
                    if pred == last_pred:
                        held_time = current_time - start_time
                        if held_time >= 3:
                            # Accept sign → update buffer
                            if pred == "DEL":
                                text_buffer = text_buffer[:-1]
                            elif pred == "SPACE":
                                text_buffer += " "
                            else:
                                text_buffer += pred

                            freeze = True
                            freeze_start = current_time
                    else:
                        last_pred = pred
                        start_time = current_time

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2)
                )

    # End freeze after 2s
    if freeze and (time.time() - freeze_start >= freeze_duration):
        freeze = False
        last_pred = None  # reset so next sign can be accepted

    # Overlay text buffer
    overlay = frame.copy()
    cv2.rectangle(overlay, (40, 20), (600, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, text_buffer if text_buffer else "...", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign to Text", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save frame snapshot when pressing 's'
        folder = "dataset/snapshots"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{int(time.time())}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

    elif key == ord('p'):  # speak out only with 'p'
        if text_buffer:
            engine.say(text_buffer)
            engine.runAndWait()

    elif key == ord('q'):  # quit with 'q'
        break

cap.release()
cv2.destroyAllWindows()
