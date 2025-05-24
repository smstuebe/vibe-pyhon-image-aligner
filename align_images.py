import os
import cv2
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_alignment.log'),
        logging.StreamHandler()
    ]
)

def align_images(images_folder, output_folder):
    """
    Aligns a sequence of images based on feature matching and saves them to the output folder.
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
    
    # Save reference image to output folder
    ref_output_path = os.path.join(output_folder, image_files[0])
    cv2.imwrite(ref_output_path, reference_img)
    logging.info(f"Saved reference image: {os.path.basename(ref_output_path)}")
    
    # Convert reference image to grayscale for feature detection
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector (SIFT works well for feature detection)
    feature_detector = cv2.SIFT_create()
    
    # Detect features in reference image
    reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(reference_gray, None)
      # Create feature matcher
    matcher = cv2.BFMatcher()
    
    # Process each image after the reference
    for i in range(1, len(image_files)):
        current_path = os.path.join(images_folder, image_files[i])
        logging.info(f"Processing image {i}/{len(image_files)-1}: {image_files[i]}...")
        
        # Read current image
        current_img = cv2.imread(current_path)
        if current_img is None:
            logging.error(f"Could not read image: {current_path}")
            continue
          # Convert to grayscale
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        current_keypoints, current_descriptors = feature_detector.detectAndCompute(current_gray, None)
        if len(current_keypoints) < 10:
            logging.warning(f"Too few keypoints detected in {image_files[i]}, skipping alignment.")
            # Save unmodified image
            output_path = os.path.join(output_folder, image_files[i])
            cv2.imwrite(output_path, current_img)
            continue
          # Match features between reference and current image
        matches = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # Need at least 4 good matches to estimate homography
        if len(good_matches) < 4:
            logging.warning(f"Not enough good matches in {image_files[i]}, skipping alignment.")
            # Save unmodified image
            output_path = os.path.join(output_folder, image_files[i])
            cv2.imwrite(output_path, current_img)
            continue
        
        # Extract location of good matches
        ref_points = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        current_points = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
          # Find homography
        H, mask = cv2.findHomography(current_points, ref_points, cv2.RANSAC, 5.0)
        
        # Apply transformation
        h, w = reference_img.shape[:2]
        aligned_img = cv2.warpPerspective(current_img, H, (w, h))
        # Save aligned image
        output_path = os.path.join(output_folder, image_files[i])
        cv2.imwrite(output_path, aligned_img)
        
        logging.info(f"Saved aligned image: {os.path.basename(output_path)}")
    
    logging.info("Image alignment complete!")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    images_folder = base_dir / "img"
    output_folder = base_dir / "processed"
    
    logging.info(f"Aligning images from '{images_folder}' to '{output_folder}'")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Looking for images in: {os.path.abspath(str(images_folder))}")
    logging.info(f"Image folder exists: {os.path.exists(str(images_folder))}")
    
    if os.path.exists(str(images_folder)):
        logging.info(f"Contents of image folder: {os.listdir(str(images_folder))}")
    
    align_images(str(images_folder), str(output_folder))
