import cv2
import mediapipe as mp
import pandas as pd
import os
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def calc_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

data, labels = [], []
raw_path = "dataset/raw_images/"

# Include all letters A–Z plus DEL and SPACE
classes = [chr(c) for c in range(ord("A"), ord("Z")+1)]
classes.extend(["DEL", "SPACE"])

for cls in classes:
    folder = os.path.join(raw_path, cls)
    if not os.path.exists(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                scale = calc_distance(wrist, middle_mcp) or 1e-6

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([(lm.x - wrist.x) / scale,
                                (lm.y - wrist.y) / scale,
                                (lm.z - wrist.z) / scale])
                data.append(row)
                labels.append(cls)

df = pd.DataFrame(data)
df['label'] = labels
os.makedirs("dataset/landmarks", exist_ok=True)
df.to_csv("dataset/landmarks/landmarks.csv", index=False)

print(" Landmarks saved for classes:", classes)
