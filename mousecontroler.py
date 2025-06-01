import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

while True:
    ret, image = webcam.read()
    if not ret:
        print("Error: Couldn't read image from webcam.")
        break

    window_h, window_w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    process_img = face_mesh.process(rgb)
    all_face_landmarks_points = process_img.multi_face_landmarks

    if all_face_landmarks_points:  # Ensure landmarks are detected
        for face_landmarks in all_face_landmarks_points:
            one_face_landmarks_points = face_landmarks.landmark

            # Check if sufficient landmarks are present
            if len(one_face_landmarks_points) >= 478:
                # Draw specific landmarks
                for landmark_points in one_face_landmarks_points[474:478]:
                    x = int(landmark_points.x * window_w)
                    y = int(landmark_points.y * window_h)
                    cv2.circle(img=image, center=(x, y), radius=1, color=(0, 0, 255), thickness=1)

                left_eye = [one_face_landmarks_points[145], one_face_landmarks_points[155]]
                
                # Calculate the center of the left eye
                left_eye_x = int((left_eye[0].x + left_eye[1].x) / 2 * window_w)
                left_eye_y = int((left_eye[0].y + left_eye[1].y) / 2 * window_h)

                # Draw the center of the left eye
                cv2.circle(img=image, center=(left_eye_x, left_eye_y), radius=5, color=(0, 255, 255), thickness=-1)

                # Map eye position to screen coordinates
                cursor_x = int(left_eye_x * screen_width / window_w)
                cursor_y = int(left_eye_y * screen_height / window_h)

                # Move the cursor
                pyautogui.moveTo(cursor_x, cursor_y)

                # Optional: Click if the eye is closed (implement your own logic)
                if abs(left_eye[0].y - left_eye[1].y) < 0.01:
                    pyautogui.click()
                    pyautogui.sleep(2)
                    print("Mouse click")

    cv2.imshow("window", image)

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to exit
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
