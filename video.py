import cv2
import time
import math
import numpy as np
import os
import pose
import json

WINDOW_NAME = "Visual"

# We want a easily visible bold font
preferred_font = cv2.FONT_HERSHEY_SIMPLEX

def write_to_center(frame, lines, font_size=0.5, padding=10, color=(255, 255, 255), thickness=1):
    """
    Writes each line to the center of the frame.
    """
    text_sizes = []
    for line in lines:
        text_size = cv2.getTextSize(line, preferred_font, font_size, 1)[0]
        text_sizes.append(text_size)
    total_height = sum([text_size[1] for text_size in text_sizes]) + padding * (len(lines) - 1)
    text_y = frame.shape[0] // 2 - total_height // 2 + padding
    for i, line in enumerate(lines):
        text_x = int((frame.shape[1] - text_sizes[i][0]) / 2)
        text_y += text_sizes[i][1] + padding
        cv2.putText(frame, line, (text_x, text_y), preferred_font, font_size, color, thickness, cv2.LINE_AA)

def write_to_top_left(frame, text, font_size=0.5, color=(255, 255, 255)):
    """
    Writes text to the top left corner of the frame.
    """
    text_size = cv2.getTextSize(text, preferred_font, font_size, 1)[0]
    text_x = 5
    text_y = text_size[1] + 5
    cv2.putText(frame, text, (text_x, text_y), preferred_font, font_size, color, 1, cv2.LINE_AA)

def await_keys(keys):
    """
    Waits for the user to press one of the keys on a cv2 window.
    Returns the key that was pressed.
    """
    key_ords = [ord(key) for key in keys]
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in key_ords:
            key_index = key_ords.index(key)
            return keys[key_index]

def show_shutter_effect(window_name, frame_size, length=0.5):
    """
    Shows a shutter effect on the window for length seconds.
    """
    frame = np.zeros(frame_size, dtype=np.uint8)
    frame[:] = (255, 255, 255)
    cv2.imshow(window_name, frame)
    cv2.waitKey(int(length * 1000))

def initialize(num_examples):
    vid = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    ret, frame = vid.read()
    similar_but_blank = np.zeros_like(frame) 

    instructions = f"""
    When you start, you will see a countdown timer. When it reaches 0, your pose will be captured and used as an example.
    On the first picture, you will be given 5 seconds to get back and into the pose, but the subsequent {num_examples-1} countdowns will be 3 seconds.
    After the session is finished, you should review the examples and decide which ones are worth keeping.
    If you would like to see a preview of the pose, press 'p'.
    If you would like to see a preview of you video feed in order to position yourself, press 'v'.
    Press Enter to begin...
    """

    write_to_center(similar_but_blank, instructions.split('\n'), font_size=0.5)
    write_to_top_left(similar_but_blank, f"Number of examples left: {num_examples}", font_size=0.5)
    cv2.imshow(WINDOW_NAME, similar_but_blank)
    cv2.waitKey(1)
    return vid, WINDOW_NAME

def inform_writing_frames(vid, window_name):
    """
    Shows a message saying to wait patiently for poses to be written to disk
    """
    similar_but_blank = np.zeros_like(vid.read()[1])
    message = f"""
    Performing pose estimation.
    Please wait patiently for the process to finish.
    You can see progress in the terminal.
    """
    write_to_center(similar_but_blank, message.split("\n"), font_size=0.5)
    cv2.imshow(window_name, similar_but_blank)
    cv2.waitKey(1)

def debrief(vid, window_name, save_location):
    """
    Shows a short thank you message and reminds the user to manually check the examples to ensure they are accurate.
    """
    similar_but_blank = np.zeros_like(vid.read()[1])
    message = f"""
    Thank you for participating!
    You examples are in {save_location}.
    Please manually check them to ensure accuracy.
    Press Enter to exit.
    """
    write_to_center(similar_but_blank, message.split("\n"), font_size=0.5)
    cv2.imshow(window_name, similar_but_blank)
    cv2.waitKey(1)

def preview_video(vid, window_name, break_keys=['\n', '\r']):
    """
    Shows a preview of the video feed until enter is pressed
    """
    key_ords = [ord(key) for key in break_keys]
    while True:
        ret, frame = vid.read()
        write_to_center(frame, ["Press Enter to exit preview"], font_size=2, color=(0, 0, 0))
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in key_ords:
            return

def preview_image(frame, window_name, image_path, break_keys=['\n', '\r']):
    """
    Shows a preview of the image until enter is pressed
    """
    key_ords = [ord(key) for key in break_keys]
    frame = cv2.imread(image_path)
    while True:
        write_to_center(frame, ["Press Enter to exit preview"], font_size=2, color=(0, 0, 0))
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in key_ords:
            return

def write_poses(video_buffers, save_location):
    """
    Writes the pose to the correct
    """
    existing_files = os.listdir(save_location)
    # Examples are saved in folders with indexed names. We want the first name we write here to be one greater than the last index.
    last_index = 0
    for file in existing_files:
        last_index = max(last_index, int(file))
    last_index += 1
    for i, buffer in enumerate(video_buffers):
        print("Analyzing example {}/{}".format(i+1, len(video_buffers)))
        cur_index = last_index + i
        example_dir_path = os.path.join(save_location, str(cur_index))
        os.makedirs(example_dir_path)
        last_frame = buffer[-1]
        cv2.imwrite(os.path.join(example_dir_path, 'original.jpg'), last_frame)
        # Do pose analysis
        results, annotated_frame = pose.analyze_buffer(buffer)
        cv2.imwrite(os.path.join(example_dir_path, 'annotated.jpg'), annotated_frame)
        # Write results to json file
        with open(os.path.join(example_dir_path, 'results.json'), 'w') as f:
            json.dump(results, f)
    print("Done with analysis")


def capture_pose(vid, window_name, examples_left, countdown_time=3):
    """
    Starts capturing video and counts down from countdown_time while capturing video.
    After the capture is done, it runs pose estimation on the video and returns the result from the frame when the countdown reached 0.
    """
    video_buffer = []
    start_time = time.time()
    while True:
        count = math.ceil(countdown_time - (time.time() - start_time))
        ret, frame = vid.read()
        video_buffer.append(frame.copy())
        write_to_top_left(frame, f"Number of examples left: {examples_left}", font_size=1, color=(0, 0, 255))
        write_to_center(frame, str(count), font_size=7, color=(0, 0, 255), thickness=5)
        cv2.imshow(window_name, frame)
        if count < 1:
            show_shutter_effect(window_name, frame.shape, 0.3)
            return video_buffer
        cv2.waitKey(1)
    

