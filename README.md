# Image Alignment and Timelapse Generator

***This project is 100% NOT vibe coded. Every line was written with purpose, determination, and definitely no AI assistance whatsoever... wink wink. ;)***

## Overview

This project provides a complete solution for creating professional timelapses from a series of images. It handles two critical parts of the timelapse creation process:

1. **Image Alignment** - Ensures all frames are perfectly aligned using feature detection algorithms
2. **Video Generation** - Creates smooth, professional-looking timelapse videos

Perfect for plant growth documentation, construction progress, sky changes, and any other sequential photography project.

## Features

### Image Alignment (`align_images.py`)

- Automatically aligns a sequence of images using SIFT feature detection
- Resizes images to HD resolution (720p) for optimal processing and output
- Uses parallel processing to utilize all available CPU cores
- Adds day labels to track progress over time
- Handles edge cases with graceful fallbacks

### Timelapse Generation (`create_timelapse.py`)

- Creates standard frame-by-frame timelapse videos
- Generates professional smooth transition videos with fade effects
- Configurable frame rates and transition parameters
- Multiple output video options

## Requirements

- Python 3.6+
- OpenCV
- NumPy

All dependencies can be installed using the included `requirements.txt` file.

## Setup

1. Clone this repository
2. Run the setup script to create the necessary environment:

```powershell
.\setup.ps1
```

3. Place your sequence of images in the `img` folder (jpg, jpeg, or png)

## Usage

### Step 1: Align Images

```powershell
python align_images.py
```

This will:
- Read images from the `img` folder
- Align them based on the first image
- Add day labels to track progress
- Save aligned images to the `processed` folder

### Step 2: Create Timelapse

```powershell
python create_timelapse.py
```

This will generate two video files:
- `timelapse.mp4` - Standard timelapse
- `timelapse_smooth.mp4` - Enhanced version with smooth transitions

## Customization

Both scripts have parameters that can be adjusted:

- Image resolution
- Label font size and position
- Frame rates
- Transition effects
- And more...

## How It Works

The alignment algorithm uses SIFT (Scale-Invariant Feature Transform) to detect unique features in each image, which are then matched to find correspondences between images. A homography transformation is applied to align subsequent images with the reference image.

The timelapse generator uses OpenCV to read the aligned images and convert them into video format. For the smooth transition version, it creates intermediate frames using alpha blending between consecutive images.

## License

Use however you want. Just don't blame me if your plants die while you're busy aligning photos of them.

---

*Remember: The best timelapses are the ones where you actually remember to take pictures regularly.*
