import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize video capture and Face Mesh with refined landmarks.
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Flags for left eye blink (left click).
left_blink_detected = False

# Variables for right eye blink detection.
right_blink_detected = False   # To mark a blink transition.
right_blink_counter = 0        # Count right-eye blinks.
last_right_blink_time = 0      # Time of the last right blink.
RIGHT_BLINK_TIMEOUT = 0.7      # Seconds within which a second blink is detected.

# Blink threshold (calibrate as needed).
blink_threshold = 0.015  # Example value.

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror the frame.
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    current_time = time.time()

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # ----- Cursor Movement -----
        # Use right iris landmarks (indices 474-477) to compute the center.
        iris_points = landmarks[474:478]
        iris_center_x = int(np.mean([p.x for p in iris_points]) * frame_w)
        iris_center_y = int(np.mean([p.y for p in iris_points]) * frame_h)
        cv2.circle(frame, (iris_center_x, iris_center_y), 3, (0, 255, 0), -1)
        screen_x = int(screen_w * iris_center_x / frame_w)
        screen_y = int(screen_h * iris_center_y / frame_h)
        pyautogui.moveTo(screen_x, screen_y)

        # ----- Blink Detection -----
        # Left eye: landmarks 145 and 159.
        left_eye_distance = abs(landmarks[145].y - landmarks[159].y)
        # Right eye: landmarks 374 and 386.
        right_eye_distance = abs(landmarks[374].y - landmarks[386].y)
        print(f"Left eye distance: {left_eye_distance:.5f}, Right eye distance: {right_eye_distance:.5f}")

        # Left Eye Blink (left click).
        if left_eye_distance < blink_threshold and not left_blink_detected:
            print("Left blink detected, executing left click")
            pyautogui.click(button='left')
            left_blink_detected = True
            time.sleep(0.5)  # Debounce.
        elif left_eye_distance >= blink_threshold:
            left_blink_detected = False

        # Right Eye Blink Detection (single vs. double click)
        if right_eye_distance < blink_threshold and not right_blink_detected:
            right_blink_detected = True
            if right_blink_counter == 0:
                right_blink_counter = 1
                last_right_blink_time = current_time
            elif right_blink_counter == 1:
                if current_time - last_right_blink_time < RIGHT_BLINK_TIMEOUT:
                    print("Right double blink detected, executing double click")
                    pyautogui.doubleClick(button='right')
                    right_blink_counter = 0
                else:
                    right_blink_counter = 1
                    last_right_blink_time = current_time
        elif right_eye_distance >= blink_threshold:
            right_blink_detected = False

        if right_blink_counter == 1 and (current_time - last_right_blink_time) > RIGHT_BLINK_TIMEOUT:
            print("Right single blink detected, executing right click")
            pyautogui.click(button='right')
            right_blink_counter = 0

        # ----- Optional: Visualize Blink Landmarks -----
        left_points = [(int(landmarks[145].x * frame_w), int(landmarks[145].y * frame_h)),
                       (int(landmarks[159].x * frame_w), int(landmarks[159].y * frame_h))]
        right_points = [(int(landmarks[374].x * frame_w), int(landmarks[374].y * frame_h)),
                        (int(landmarks[386].x * frame_w), int(landmarks[386].y * frame_h))]
        for point in left_points:
            cv2.circle(frame, point, 3, (255, 0, 0), -1)
        for point in right_points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

    cv2.imshow('Eye Control Cursor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
