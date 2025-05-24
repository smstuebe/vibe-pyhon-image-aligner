import cv2
import os
import glob
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timelapse.log'),
        logging.StreamHandler()
    ]
)

def create_timelapse(images_folder, output_video_path, fps=2):
    """
    Create a timelapse video from a sequence of images
    
    Args:
        images_folder: Path to folder containing processed, aligned images
        output_video_path: Path where the output video will be saved
        fps: Frames per second for the output video
    """
    # Get sorted list of image files
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpeg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    
    if not image_files:
        logging.error("No images found in the input folder.")
        return False
    
    logging.info(f"Found {len(image_files)} images to include in timelapse.")
    
    # Read first image to determine dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        logging.error(f"Could not read first image: {image_files[0]}")
        return False
    
    height, width, layers = first_img.shape
    size = (width, height)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # Process each image
    for i, file in enumerate(image_files):
        img = cv2.imread(file)
        if img is None:
            logging.warning(f"Could not read image: {file}, skipping.")
            continue
        
        # Add frame to video
        out.write(img)
        logging.info(f"Added frame {i+1}/{len(image_files)} to video")
    
    # Release video writer
    out.release()
    logging.info(f"Timelapse video saved to: {output_video_path}")
    return True

def create_slow_motion_transition(images_folder, output_video_path, fps=30, transition_frames=30):
    """
    Create a timelapse video with smooth transitions between frames
    
    Args:
        images_folder: Path to folder containing processed, aligned images
        output_video_path: Path where the output video will be saved
        fps: Frames per second for the output video
        transition_frames: Number of frames to generate for each transition
    """
    # Get sorted list of image files
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpeg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    
    if not image_files:
        logging.error("No images found in the input folder.")
        return False
    
    logging.info(f"Found {len(image_files)} images to include in timelapse with transitions.")
    
    # Read first image to determine dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        logging.error(f"Could not read first image: {image_files[0]}")
        return False
    
    height, width, layers = first_img.shape
    size = (width, height)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # Process each image and create transitions
    prev_img = None
    
    for i, file in enumerate(image_files):
        img = cv2.imread(file)
        if img is None:
            logging.warning(f"Could not read image: {file}, skipping.")
            continue
        
        # For first image, no transition needed
        if prev_img is None:
            # Add current frame multiple times to give time to see first image
            for _ in range(int(fps * 1.5)):  # 1.5 seconds on first frame
                out.write(img)
        else:
            # Create transition frames
            for j in range(transition_frames):
                # Calculate alpha for blending (0.0 -> 1.0)
                alpha = j / transition_frames
                # Blend images
                transition = cv2.addWeighted(prev_img, 1 - alpha, img, alpha, 0)
                out.write(transition)
            
            # Hold the current frame for a moment
            for _ in range(int(fps * 1)):  # 1 second on each frame
                out.write(img)
        
        prev_img = img.copy()
        logging.info(f"Processed image {i+1}/{len(image_files)}")
    
    # Hold the last frame a bit longer
    for _ in range(int(fps * 2)):  # 2 seconds on last frame
        out.write(prev_img)
    
    # Release video writer
    out.release()
    logging.info(f"Smooth transition video saved to: {output_video_path}")
    return True

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    images_folder = base_dir / "processed"
    
    # Create standard timelapse
    output_standard = base_dir / "tomato_timelapse.mp4"
    create_timelapse(str(images_folder), str(output_standard), fps=1)
    
    # Create smooth transition timelapse
    output_smooth = base_dir / "tomato_timelapse_smooth.mp4"
    create_slow_motion_transition(str(images_folder), str(output_smooth), fps=30, transition_frames=60)
