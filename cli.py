from pyfiglet import Figlet
import os
from config import POSES
import video

f = Figlet(font='slant')

def snake_to_title(snake):
    """
    Converts a snake_case string to a Title Case string
    This means turning each _ to a space and capitalizing the first letter of each word
    """
    return snake.title().replace('_', ' ')

def main():
    print(f.renderText('Why did I spend time on this?'))
    os.makedirs('./data', exist_ok=True)

    example_counts = {}
    for pose in POSES:
        os.makedirs(f'./data/{pose}', exist_ok=True)
        example_counts[pose] = len(os.listdir(f'./data/{pose}'))

    prompt = "Which pose would you like to take more examples of? (Enter the number)\n"
    for i, pose in enumerate(POSES):
        prompt += f"{i+1}. {snake_to_title(pose)} (Existing example count: {example_counts[pose]})\n"

    try:
        pose_number = int(input(prompt))
    except ValueError:
        raise ValueError("Please enter a valid number")
    if pose_number < 1 or pose_number > len(POSES):
        raise ValueError("Please enter a valid number")
    
    pose = POSES[pose_number-1]

    data_dir = f'./data/{pose}'
    example_image = f'./pose_definitions/{pose}.jpg'
    if not os.path.exists(example_image):
        raise RuntimeError(f"Could not find example image at {example_image}")

    print(f"You selected {snake_to_title(pose)}.")
    num_examples = int(input("How many examples would you like to take? (Enter the number)\n"))
    print(f"Recording {num_examples} examples of {snake_to_title(pose)}. Find the camera window to begin.")
    ready = False
    while not ready:
        vid, window_name = video.initialize(num_examples)
        key = video.await_keys(['p', 'v', '\n', '\r'])
        if key == 'p':
            # Should show a preview of the pose
            video.preview_image(example_image, window_name, example_image)
        elif key == 'v':
            # Should show a preview of the video feed
            video.preview_video(vid, window_name)
        elif key in ['\n', '\r']:
            ready = True
    examples_left = num_examples
    example_frames = []
    while examples_left > 0:
        countdown_time = 3 if examples_left != num_examples else 5
        frames = video.capture_pose(vid, window_name, examples_left, countdown_time)
        example_frames.append(frames)
        examples_left -= 1
    video.inform_writing_frames(vid, window_name)
    video.write_poses(example_frames, data_dir)
    video.debrief(vid, window_name, f'./data/{pose}')
    video.await_keys(['\n', '\r'])

if __name__ == '__main__':
    main()