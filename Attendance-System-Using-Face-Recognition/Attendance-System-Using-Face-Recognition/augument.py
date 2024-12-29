import cv2
import numpy as np
from pathlib import Path
import dlib
from tqdm import tqdm
from albumentations import (
    Compose, RandomBrightnessContrast, GaussNoise, 
    HorizontalFlip, Rotate, RandomGamma, Blur
)

def setup_augmentation():
    """Setup the augmentation pipeline"""
    return Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.7),
        GaussNoise(p=0.3),
        Rotate(limit=15, p=0.5),
        RandomGamma(p=0.3),
        Blur(blur_limit=3, p=0.3),
    ])

def process_single_folder(input_folder: Path, output_folder: Path, augmentations_per_image: int = 5):
    """
    Process images in a single folder and create augmented versions
    
    Args:
        input_folder: Path to folder containing original images
        output_folder: Path to save augmented dataset
        augmentations_per_image: Number of augmented versions to create per original image
    """
    augmentor = setup_augmentation()
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))
    
    image_counter = 1
    for img_path in tqdm(image_files, desc=f"Processing {input_folder.name}"):
        try:
            # Read and process original image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Resize to standard size
            image = cv2.resize(image, (256, 256))
            
            # Save original image
            cv2.imwrite(str(output_folder / f"{image_counter}.jpg"), image)
            image_counter += 1
            
            # Generate augmented versions
            for _ in range(augmentations_per_image):
                augmented = augmentor(image=image)['image']
                cv2.imwrite(str(output_folder / f"{image_counter}.jpg"), augmented)
                image_counter += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return image_counter - 1

def batch_augment_dataset(input_base_path: str, output_base_path: str, augmentations_per_image: int = 5):
    """
    Process all person folders and create augmented dataset
    
    Args:
        input_base_path: Path to base directory containing person folders
        output_base_path: Path to save augmented dataset
        augmentations_per_image: Number of augmented versions to create per original image
    """
    input_base = Path(input_base_path)
    output_base = Path(output_base_path)
    
    if not input_base.exists():
        raise ValueError(f"Input directory {input_base} does not exist")
    
    # Create output base directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each person's folder
    total_images = 0
    person_folders = [d for d in input_base.iterdir() if d.is_dir()]
    
    print(f"Found {len(person_folders)} person folders")
    
    for person_folder in person_folders:
        print(f"\nProcessing folder: {person_folder.name}")
        output_folder = output_base / person_folder.name
        
        images_processed = process_single_folder(
            person_folder,
            output_folder,
            augmentations_per_image
        )
        
        total_images += images_processed
        print(f"Created {images_processed} images for {person_folder.name}")
    
    print(f"\nProcessing complete!")
    print(f"Total images in augmented dataset: {total_images}")
    return total_images

if __name__ == "__main__":
    # Example usage
    INPUT_PATH = "E:\Attendance-Management-system-using-face-recognition-master\Attendance-Management-system-using-face-recognition-master\TrainingImage"  # Base folder containing person folders
    OUTPUT_PATH = "E:\Attendance-Management-system-using-face-recognition-master\Attendance-Management-system-using-face-recognition-master\AUG"  # Where to save augmented dataset
    AUGMENTATIONS_PER_IMAGE = 5  # Number of augmented versions per original image
    
    try:
        total_images = batch_augment_dataset(
            INPUT_PATH,
            OUTPUT_PATH,
            AUGMENTATIONS_PER_IMAGE
        )
        
        print("\nSummary:")
        print(f"Input directory: {INPUT_PATH}")
        print(f"Output directory: {OUTPUT_PATH}")
        print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
        print(f"Total images created: {total_images}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")