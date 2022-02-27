"""
Module for doing the pose estimation
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def landmarks_list_to_array(results):
    """
    Converts a list of landmarks to an array
    """
    keys_points = []
    for landmark in results.pose_landmarks.landmark:
        keys_points.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    return keys_points

def analyze_buffer(buffer):
    """
    Buffer contains cv2 frames to be analyzed as a sequence.
    The pose of the last image should be returned as well as an annotated image of the last frame.
    """
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        for frame in buffer:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            if not results.pose_landmarks:
                continue
        
        # When we get here, all frames have been processed and the final frame and results are available.
        # We can now draw the pose on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return landmarks_list_to_array(results), frame