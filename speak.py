import cv2
import pyttsx3
import threading
import os
import time

text_buffer = "Hello this is my sign language project"

cap = cv2.VideoCapture(0)

p_pressed = False
s_pressed = False


# -------- Threaded Speak Function --------
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


print("P = Speak | S = Snapshot | Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame,
                "P = Speak | S = Snapshot | Q = Quit",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow("Speak Window", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- SPEAK --------
    if key == ord('p') and not p_pressed:
        p_pressed = True

        if text_buffer:
            print("Speaking:", text_buffer)

            # Run speech in separate thread
            threading.Thread(target=speak_text, args=(text_buffer,)).start()

    elif key != ord('p'):
        p_pressed = False

    # -------- SNAPSHOT --------
    if key == ord('s') and not s_pressed:
        s_pressed = True

        os.makedirs("snapshots", exist_ok=True)
        filename = f"snapshots/{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print("Snapshot saved:", filename)

    elif key != ord('s'):
        s_pressed = False

    # -------- QUIT --------
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()