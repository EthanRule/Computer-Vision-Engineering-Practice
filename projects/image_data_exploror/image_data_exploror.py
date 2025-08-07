import os
from pathlib import Path
import pandas as pd
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_dir = Path(r"D:\Repos\Computer-Vision-Engineering-Practice\Projects\image_data_exploror\images")

image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

if not image_files:
    print("No images found in the file dir")

print(image_files)

data = {
    "file_name": [],
    "width": [],
    "height": [],
    "mode": [],
}

# Store images for histogram processing
images_data = []

for image_path in image_files:
    img = cv.imread(str(image_path))

    if img is None:
        print(f"failed to read image: {image_path.name}")
        continue
    
    height, width = img.shape[:2]

    if len(img.shape) == 2:
        mode = "grayscale"
    elif len(img.shape) == 3 and img.shape[2] == 3:
        mode = "rgb"
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        images_data.append((image_path.name, img_rgb))

    data["file_name"].append(image_path.name)
    data["width"].append(width)
    data["height"].append(height)
    data["mode"].append(mode)

df = pd.DataFrame(data)
print(df)

# Plot RGB histograms for each image
num_images = len(images_data)
if num_images > 0:
    # Calculate subplot dimensions
    cols = min(3, num_images)  # Max 3 columns
    rows = (num_images + cols - 1) // cols  # Calculate required rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    
    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')
    
    for idx, (filename, img_rgb) in enumerate(images_data):
        if rows > 1:
            ax = axes[idx // cols][idx % cols]
        else:
            ax = axes[idx] if cols > 1 else axes
        
        # Calculate histogram for each channel
        for i, color in enumerate(colors):
            histogram = cv.calcHist([img_rgb], [i], None, [256], [0, 256])
            ax.plot(histogram, color=color, label=channel_names[i])
        
        ax.set_title(f'RGB Histogram - {filename}')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    if rows > 1:
        for idx in range(num_images, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Alternative: Plot histograms using matplotlib.pyplot.hist() instead of cv.calcHist()
if num_images > 0:
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    
    for idx, (filename, img_rgb) in enumerate(images_data):
        if rows > 1:
            ax = axes[idx // cols][idx % cols]
        else:
            ax = axes[idx] if cols > 1 else axes
        
        # Extract each channel and plot histogram using pyplot.hist()
        red_channel = img_rgb[:, :, 0].flatten()
        green_channel = img_rgb[:, :, 1].flatten()
        blue_channel = img_rgb[:, :, 2].flatten()
        
        ax.hist(red_channel, bins=256, color='red', alpha=0.5, label='Red', range=[0, 256])
        ax.hist(green_channel, bins=256, color='green', alpha=0.5, label='Green', range=[0, 256])
        ax.hist(blue_channel, bins=256, color='blue', alpha=0.5, label='Blue', range=[0, 256])
        
        ax.set_title(f'RGB Histogram - {filename}')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    if rows > 1:
        for idx in range(num_images, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
    
    plt.tight_layout()
    plt.show()

