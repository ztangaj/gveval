import cv2
import numpy as np
import json
import os


def select_prompt(visual, setting, accr=False):
    base_path = os.path.join(os.path.dirname(__file__), 'prompts')
    if visual == 'vid':
        if accr:
            return os.path.join(base_path, 'vid', 'accr', f'{setting}.txt')
        else:
            return os.path.join(base_path, 'vid', f'{setting}.txt')
    elif visual == 'img':
        return os.path.join(base_path, 'img', f'{setting}.txt')
    else:
        return None


def save_frames_as_single_image(frames, output_path='combined_image.png', num_samples=3):
    # Set the height and width of each frame
    height, width = 512, 512
    
    # Initialize a large empty image with dynamic width based on num_samples
    combined_image = np.zeros((height, width * num_samples, 3), dtype=np.uint8)
    
    # Calculate the indices of the frames to sample
    total_frames = len(frames)
    if num_samples > 1:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    else:
        indices = np.array([0])
    
    for index, frame_index in enumerate(indices):
        # Get the frame at the sampled index
        frame = frames[frame_index]
        
        # Resize frame to fit 512x462 (leaving space for the 50px annotation bar)
        frame_resized = cv2.resize(frame, (width, height - 50))  # Deduct 50 pixels for the annotation bar
        
        # Create a white bar for annotation
        bar = np.ones((50, width, 3), dtype=np.uint8) * 255
        
        # Put text on the bar
        cv2.putText(bar, f'Frame {frame_index + 1}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # Combine the resized frame and the bar
        combined_frame = np.vstack((frame_resized, bar))
        
        # Place the combined frame into the correct position in the large image
        combined_image[:, index*width:(index+1)*width] = combined_frame
    
    # Save the combined image
    cv2.imwrite(output_path, combined_image)

def video2imgs(video_path, output_path, num_samples=3, save_combined=True):
    cap = cv2.VideoCapture(video_path)
    # Check if video loaded successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        # exit()  # Do not exit the program directly
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_samples == -1:
        frame_indices = np.arange(frame_count)  # Get all frame indices
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_samples+1, dtype=int)  # Get indices for num_samples frames plus the last frame
    # Create a list to store the frames
    frames = []
    for i in frame_indices[:-1]:  # Exclude the last frame index as it's just for calculation
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Failed to retrieve frame at index {i}")
    # Release the video capture object
    cap.release()
    if save_combined:
        # Save and display the combined image
        save_frames_as_single_image(frames, output_path, num_samples)
    else:
        # Create a new folder to save individual frames
        folder_path = os.path.splitext(output_path)[0]
        os.makedirs(folder_path, exist_ok=True)
        # Save each frame as a separate image
        for index, frame in enumerate(frames):
            frame_path = os.path.join(folder_path, f"frame_{index}.png")
            cv2.imwrite(frame_path, frame)
