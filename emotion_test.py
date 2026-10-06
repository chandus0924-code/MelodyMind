import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    input("Press Enter to close...")
    exit()

print("Camera started.")
print("Press S to detect your emotion.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "Press S to scan emotion | Q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("MelodyMind Emotion Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("Analyzing your emotion...")

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]

            print(f"Detected emotion: {emotion}")
            print(f"Confidence: {confidence:.2f}%")

        except Exception as error:
            print("Emotion detection error:")
            print(error)

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
