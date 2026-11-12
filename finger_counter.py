
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark positions
finger_tips = [8, 12, 16, 20]

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():

        success, img = cap.read()

        if not success:
            print("Camera not detected")
            break

        # Mirror camera
        img = cv2.flip(img, 1)

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb_img)

        total_fingers = 0

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmarks = hand_landmarks.landmark

                # Count thumb
                if landmarks[4].x < landmarks[3].x:
                    fingers = 1
                else:
                    fingers = 0

                # Count remaining four fingers
                for tip in finger_tips:

                    if landmarks[tip].y < landmarks[tip - 2].y:
                        fingers += 1

                total_fingers += fingers

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Display finger count
        cv2.putText(
            img,
            f"Fingers: {total_fingers}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.imshow("Finger Counter", img)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
