import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe drawing and pose instances
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_distance_from_left(x):
    """Calculate the distance of the point from the left edge of the screen."""
    return x * 640  # Assuming the image width is 640 pixels


def process_video(source=0, is_file=False, box_coords=(100, 100, 200, 200), joint='LEFT_ANKLE', alert_inside=True):
    # Video capture
    cap = cv2.VideoCapture(source)

    # Setup mediapipe instance
    pose = None

    # Counter variables
    counter = 0
    in_box = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        if pose is None:
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of the specified joint
            joint_enum = getattr(mp_pose.PoseLandmark, joint)
            joint_landmark = landmarks[joint_enum.value]
            joint_coords = (joint_landmark.x, joint_landmark.y)
            joint_confidence = joint_landmark.visibility

            # Calculate distance from the left edge of the screen
            distance_from_left = calculate_distance_from_left(joint_landmark.x)

            # Visualize joint coordinates, distance, and confidence
            cv2.putText(image, f'Coords: ({joint_coords[0]:.2f}, {joint_coords[1]:.2f})',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Distance from left: {distance_from_left:.2f}px',
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Confidence: {joint_confidence:.2f}',
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw box
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if the specified joint is inside the box
            foot_in_box = x1 < joint_landmark.x * 640 < x2 and y1 < joint_landmark.y * 480 < y2

            if alert_inside and foot_in_box and not in_box:
                in_box = True
                counter += 1
            elif not alert_inside and not foot_in_box and in_box:
                in_box = False
                counter += 1

            if alert_inside and foot_in_box or not alert_inside and not foot_in_box:
                cv2.putText(image, "In Box!" if alert_inside else "Out of Box!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        except:
            # If landmarks are not detected, turn off the pose model
            if pose is not None:
                pose.close()
                pose = None

        # Render detections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display counter
        cv2.putText(image, f'Counter: {counter}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage:
# Live video feed with default LEFT_ANKLE and box (100, 100, 200, 200)
process_video(source=0, is_file=False, box_coords=(100, 100, 200, 200), joint='RIGHT_ANKLE', alert_inside=True)

# Video file input with a different joint and box coordinates
# process_video(source='path_to_video.mp4', is_file=True, box_coords=(100, 100, 200, 200), joint='LEFT_KNEE', alert_inside=False)
