import os
import cv2
import numpy as np
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_alignment.log'),
        logging.StreamHandler()
    ]
)

def get_optimal_process_count():
    return multiprocessing.cpu_count()

def resize_image(image, target_width=1280):
    """Resize image to target width while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    # Calculate scale to maintain aspect ratio but limit to target dimensions
    scale = min(target_width / w, 720 / h)
    
    # Only resize if the image is larger than target
    if scale < 1:
        new_size = (int(w*scale), int(h*scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale
    return image, 1.0

def add_day_label(image, filename, base_date="20250521"):
    """
    Add day number label to the bottom left corner of the image
    Calculates days relative to a base date (first day is Day 1)
    """
    # Extract date from filename (assuming format contains date YYYYMMDD)
    date_match = re.search(r'(\d{8})', filename)
    
    if date_match:
        date_str = date_match.group(1)
        try:
            # Extract day number from date
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            # Extract base date components
            base_year = int(base_date[:4])
            base_month = int(base_date[4:6])
            base_day = int(base_date[6:8])
            
            # Calculate day difference
            import datetime
            current_date = datetime.date(year, month, day)
            base_date_obj = datetime.date(base_year, base_month, base_day)
            day_diff = (current_date - base_date_obj).days + 1  # +1 so first day is Day 1
            
            label = f"Day {day_diff}"
        except Exception as e:
            # Fallback if date calculation fails
            logging.warning(f"Failed to calculate day number: {e}")
            label = filename
    else:
        # Fallback if no date pattern found
        label = filename
      # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)  # White color
    
    # Add black outline for better visibility
    outline_color = (0, 0, 0)
    outline_thickness = 3
    
    # Position text in bottom left with padding
    h, w = image.shape[:2]
    x_pos = 20
    y_pos = h - 30  # 30px from bottom
    
    # Draw outline by placing text multiple times with offsets
    cv2.putText(image, label, (x_pos, y_pos), font, font_scale, outline_color, outline_thickness)
    
    # Draw the main text
    cv2.putText(image, label, (x_pos, y_pos), font, font_scale, font_color, font_thickness)
    
    return image

def process_single_image(args):
    """
    Process a single image for alignment
    
    Args:
        args: Tuple containing (image_file, images_folder, output_folder, reference_img_path)
    """
    image_file, images_folder, output_folder, reference_img_path = args
    
    try:
        # Load reference image
        reference_img = cv2.imread(reference_img_path)
        
        # Resize reference image to 720p equivalent for faster processing
        reference_img, ref_scale = resize_image(reference_img)
        
        reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        feature_detector = cv2.SIFT_create()
        reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(reference_gray, None)
        
        current_path = os.path.join(images_folder, image_file)
        
        # Read current image
        current_img = cv2.imread(current_path)
        if current_img is None:
            logging.error(f"Could not read image: {current_path}")
            return False
        
        # Resize current image to same scale for consistent alignment
        current_img, curr_scale = resize_image(current_img)
        
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        current_keypoints, current_descriptors = feature_detector.detectAndCompute(current_gray, None)
        if len(current_keypoints) < 10:
            logging.warning(f"Too few keypoints detected in {image_file}, skipping alignment.")
            # Save unmodified image (but resized)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, current_img)
            return False
        
        # Create feature matcher
        matcher = cv2.BFMatcher()
        
        # Match features between reference and current image
        matches = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Need at least 4 good matches to estimate homography
        if len(good_matches) < 4:
            logging.warning(f"Not enough good matches in {image_file}, skipping alignment.")
            # Save unmodified image (but resized)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, current_img)
            return False
            
        # Extract location of good matches
        ref_points = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        current_points = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(current_points, ref_points, cv2.RANSAC, 5.0)
          # Apply transformation
        h, w = reference_img.shape[:2]
        aligned_img = cv2.warpPerspective(current_img, H, (w, h))
        
        # Add day label to the image
        aligned_img = add_day_label(aligned_img, image_file)
        
        # Save aligned image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, aligned_img)
        
        logging.info(f"Saved aligned image: {image_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {image_file}: {str(e)}")
        return False

def align_images(images_folder, output_folder):
    """
    Aligns a sequence of images based on feature matching and saves them to the output folder.
    Uses parallel processing to utilize all CPU cores.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(images_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        logging.error("No images found in the input folder.")
        return
    
    logging.info(f"Found {len(image_files)} images to process.")
    
    # Read the first image as the reference image
    reference_path = os.path.join(images_folder, image_files[0])
    reference_img = cv2.imread(reference_path)
    
    if reference_img is None:
        logging.error(f"Could not read reference image: {reference_path}")
        return
    
    # Resize reference image for output
    reference_img_resized, scale = resize_image(reference_img)
    logging.info(f"Resized reference image from {reference_img.shape[1]}x{reference_img.shape[0]} to {reference_img_resized.shape[1]}x{reference_img_resized.shape[0]}")
      # Add day label to reference image (Day 1)
    reference_img_resized = add_day_label(reference_img_resized, image_files[0])
    
    # Save reference image to output folder
    ref_output_path = os.path.join(output_folder, image_files[0])
    cv2.imwrite(ref_output_path, reference_img_resized)
    logging.info(f"Saved reference image: {os.path.basename(ref_output_path)}")
    
    # Get optimal number of workers
    num_workers = get_optimal_process_count()
    logging.info(f"Using {num_workers} workers for parallel processing")
    
    # Prepare data for parallel processing (skip the reference image)
    tasks = [(image_files[i], images_folder, output_folder, reference_path) 
             for i in range(1, len(image_files))]
    
    # Process images in parallel using optimized number of workers
    start_time = time.time()
    successful = 0
    if tasks:
        # Explicitly setting max_workers and using chunksize for better workload distribution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Use a smaller chunk size for better load balancing
            chunksize = max(1, len(tasks) // (num_workers * 2))
            results = list(executor.map(process_single_image, tasks, chunksize=chunksize))
            successful = results.count(True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Image alignment complete! Successfully aligned {successful} out of {len(tasks)} images in {processing_time:.2f} seconds.")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    images_folder = base_dir / "img"
    output_folder = base_dir / "processed"
    
    logging.info(f"Aligning images from '{images_folder}' to '{output_folder}'")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Looking for images in: {os.path.abspath(str(images_folder))}")
    
    if os.path.exists(str(images_folder)):
        logging.info(f"Contents of image folder: {os.listdir(str(images_folder))}")
    
    align_images(str(images_folder), str(output_folder))
