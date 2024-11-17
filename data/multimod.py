import os
import cv2
import numpy as np
from tqdm import tqdm

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames(video_path, output_folder, num_frames=800):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the start and end frames to extract
    if total_frames > num_frames:
        start_frame = (total_frames - num_frames) // 2
        end_frame = start_frame + num_frames
    else:
        start_frame = 0
        end_frame = total_frames
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame and i < end_frame:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
    
    cap.release()
    
    # If the video has less frames, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return frames

def save_image(image, path):
    cv2.imwrite(path, image)

def save_tensor(tensor, path):
    np.save(path, tensor)

def extract_pixels_from_views(frames, output_folder, view_type):
    height, width = frames[0].shape
    center_x, center_y = width // 2, height // 2
    
    if view_type == 0:  # Vertical center line
        pixels = [frame[:, center_x] for frame in frames]
    elif view_type == 1:  # Horizontal center line
        pixels = [frame[center_y, :] for frame in frames]
    elif view_type == 2:  # Diagonal from top-left to bottom-right
        pixels = [np.diag(frame) for frame in frames]
    elif view_type == 3:  # Diagonal from bottom-left to top-right
        pixels = [np.diag(np.fliplr(frame)) for frame in frames]
    else:
        raise ValueError("Invalid view type")
    
    # Convert list of pixels to image and tensor
    img = np.array(pixels).T  # Transpose to match required shape (height x num_frames)
    img_normalized = (img - img.min()) / (img.max() - img.min()) * 255
    tensor = img / 255.0
    
    # Save image and tensor
    save_image(img_normalized.astype(np.uint8), os.path.join(output_folder, f'{view_type}.png'))
    save_tensor(tensor, os.path.join(output_folder, f'Tensor_{view_type}.npy'))

def process_videos(input_folder, output_folder, num_frames=800):
    create_directory(output_folder)
    
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4') or f.endswith('.avi')]
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        create_directory(video_output_folder)
        
        # Extract frames from video
        frames = extract_frames(video_path, video_output_folder, num_frames)
        
        # Extract pixels from different views
        for view_type in range(4):  # Views 0, 1, 2, 3
            extract_pixels_from_views(frames, video_output_folder, view_type)

if __name__ == "__main__":
    input_folder = r"\EchoNet-Dynamic\Videos"
    output_folder = r"\Dataset"
    process_videos(input_folder, output_folder)
