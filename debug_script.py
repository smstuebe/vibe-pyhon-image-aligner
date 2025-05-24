import os
import cv2
import numpy as np
from pathlib import Path

def main():
    # Define paths
    base_dir = Path(__file__).parent
    images_folder = base_dir / "img"
    output_folder = base_dir / "processed"
    
    # Print basic debug information
    print(f"Script is running!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Images folder: {str(images_folder)}")
    print(f"Output folder: {str(output_folder)}")
    
    # Check if image folder exists
    if not os.path.exists(str(images_folder)):
        print(f"Error: Image folder not found at {str(images_folder)}")
        return
    
    # List image files
    image_files = sorted([f for f in os.listdir(str(images_folder)) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(image_files)} images: {image_files}")
    
    # Ensure output directory exists
    os.makedirs(str(output_folder), exist_ok=True)
    
    # Process a sample image
    if image_files:
        sample_image_path = os.path.join(str(images_folder), image_files[0])
        try:
            img = cv2.imread(sample_image_path)
            if img is None:
                print(f"Failed to read image: {sample_image_path}")
            else:
                print(f"Successfully read image: {sample_image_path}, size: {img.shape}")
                
                # Create a simple processed version (grayscale)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Save to output folder
                output_path = os.path.join(str(output_folder), f"gray_{image_files[0]}")
                cv2.imwrite(output_path, gray)
                print(f"Saved grayscale image to: {output_path}")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
    # Write a test file to ensure we have write permissions
    try:
        with open(os.path.join(str(base_dir), "test_output.txt"), "w") as f:
            f.write("Test successful")
        print("Successfully wrote test file")
    except Exception as e:
        print(f"Error writing test file: {str(e)}")

if __name__ == "__main__":
    try:
        main()
        print("Script completed successfully")
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
