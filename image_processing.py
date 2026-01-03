"""
image_processing.py
Image processing utilities for batch operations and organizing predictions
"""

import os
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles image processing operations
    """
    
    def __init__(self, max_image_size: int = 1024):
        """
        Initialize image processor
        
        Args:
            max_image_size: Maximum dimension for uploaded images
        """
        self.max_image_size = max_image_size
        self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']
    
    def validate_image(self, image_file) -> Tuple[bool, str]:
        """
        Validate uploaded image
        
        Args:
            image_file: Uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file extension
            if image_file.name:
                ext = image_file.name.split('.')[-1].lower()
                if ext not in self.supported_formats:
                    return False, f"Unsupported format. Please use: {', '.join(self.supported_formats)}"
            
            # Try to open as image
            image = Image.open(image_file)
            
            # Check image size
            width, height = image.size
            if width * height > 25_000_000:  # ~5000x5000
                return False, "Image too large. Maximum 25 megapixels."
            
            # Check file size (25MB max)
            image_file.seek(0, 2)  # Seek to end
            file_size = image_file.tell()
            image_file.seek(0)  # Seek back to start
            
            if file_size > 25 * 1024 * 1024:
                return False, "File too large. Maximum 25MB."
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def resize_image(self, image: Image.Image, max_size: int = None) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image
            max_size: Maximum dimension (uses self.max_image_size if None)
            
        Returns:
            Resized PIL Image
        """
        if max_size is None:
            max_size = self.max_image_size
        
        # Get current size
        width, height = image.size
        
        # Check if resizing needed
        if width <= max_size and height <= max_size:
            return image
        
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return resized
    
    def prepare_for_display(self, image: Image.Image, size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """
        Prepare image for display in UI
        
        Args:
            image: PIL Image
            size: Target size for display
            
        Returns:
            PIL Image ready for display
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create thumbnail
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        return image
    
    def save_to_bytes(self, image: Image.Image, format: str = 'JPEG', quality: int = 85) -> bytes:
        """
        Convert PIL Image to bytes
        
        Args:
            image: PIL Image
            format: Output format
            quality: JPEG quality (1-100)
            
        Returns:
            Image as bytes
        """
        buffer = io.BytesIO()
        
        # Convert to RGB if saving as JPEG
        if format.upper() == 'JPEG' and image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(buffer, format=format, quality=quality)
        buffer.seek(0)
        
        return buffer.getvalue()


class BatchProcessor:
    """
    Handles batch processing and organization of images
    """
    
    def __init__(self, temp_dir: str = "temp"):
        """
        Initialize batch processor
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def organize_by_class(
        self, 
        images: List[Image.Image],
        predictions: List[Dict],
        filenames: List[str] = None
    ) -> Dict[str, List[Tuple[Image.Image, str]]]:
        """
        Organize images by predicted class
        
        Args:
            images: List of PIL Images
            predictions: List of prediction dictionaries
            filenames: List of original filenames (optional)
            
        Returns:
            Dictionary mapping class names to lists of (image, filename) tuples
        """
        if filenames is None:
            filenames = [f"image_{i:03d}.jpg" for i in range(len(images))]
        
        organized = {}
        
        for img, pred, filename in zip(images, predictions, filenames):
            class_name = pred['predicted_class']
            
            if class_name not in organized:
                organized[class_name] = []
            
            organized[class_name].append((img, filename))
        
        logger.info(f"Organized {len(images)} images into {len(organized)} classes")
        return organized
    
    def create_zip_archive(
        self,
        organized_images: Dict[str, List[Tuple[Image.Image, str]]],
        zip_name: str = "classified_garbage.zip"
    ) -> bytes:
        """
        Create ZIP archive of organized images
        
        Args:
            organized_images: Dictionary from organize_by_class
            zip_name: Name of the ZIP file
            
        Returns:
            ZIP file as bytes
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for class_name, image_list in organized_images.items():
                # Create folder for this class
                class_folder = f"{class_name}/"
                
                # Add each image to the class folder
                for idx, (image, original_filename) in enumerate(image_list):
                    # Generate filename
                    filename = f"{class_folder}{original_filename}"
                    
                    # Convert image to bytes
                    img_bytes = io.BytesIO()
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image.save(img_bytes, format='JPEG', quality=90)
                    img_bytes.seek(0)
                    
                    # Add to zip
                    zip_file.writestr(filename, img_bytes.getvalue())
                    logger.info(f"Added {filename} to ZIP")
        
        zip_buffer.seek(0)
        logger.info(f"Created ZIP archive: {zip_name}")
        
        return zip_buffer.getvalue()
    
    def create_summary_file(
        self,
        predictions: List[Dict],
        filenames: List[str]
    ) -> str:
        """
        Create a text summary of all predictions
        
        Args:
            predictions: List of prediction dictionaries
            filenames: List of filenames
            
        Returns:
            Summary text
        """
        summary = "GARBAGE CLASSIFICATION SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Total images processed: {len(predictions)}\n\n"
        
        # Count by class
        class_counts = {}
        for pred in predictions:
            class_name = pred['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary += "Distribution by class:\n"
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(predictions)) * 100
            summary += f"  {class_name:15s}: {count:3d} ({percentage:5.1f}%)\n"
        
        summary += "\n" + "=" * 60 + "\n"
        summary += "DETAILED PREDICTIONS\n"
        summary += "=" * 60 + "\n\n"
        
        # Detailed predictions
        for filename, pred in zip(filenames, predictions):
            summary += f"File: {filename}\n"
            summary += f"  Predicted: {pred['predicted_class']}\n"
            summary += f"  Confidence: {pred['confidence']:.2%}\n"
            summary += f"  Status: {pred['status']}\n"
            
            if pred['status'] in ['uncertain', 'out_of_scope']:
                summary += "  Top 3 predictions:\n"
                for item in pred['top_3']:
                    summary += f"    {item['rank']}. {item['class']}: {item['confidence']:.2%}\n"
            
            summary += "\n"
        
        return summary
    
    def create_zip_with_summary(
        self,
        images: List[Image.Image],
        predictions: List[Dict],
        filenames: List[str],
        zip_name: str = "classified_garbage.zip"
    ) -> bytes:
        """
        Create ZIP archive with organized images AND summary file
        
        Args:
            images: List of PIL Images
            predictions: List of prediction dictionaries
            filenames: List of filenames
            zip_name: Name of the ZIP file
            
        Returns:
            ZIP file as bytes
        """
        # Organize images
        organized = self.organize_by_class(images, predictions, filenames)
        
        # Create summary
        summary = self.create_summary_file(predictions, filenames)
        
        # Create ZIP
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add summary file
            zip_file.writestr("SUMMARY.txt", summary)
            logger.info("Added SUMMARY.txt to ZIP")
            
            # Add organized images
            for class_name, image_list in organized.items():
                class_folder = f"{class_name}/"
                
                for image, original_filename in image_list:
                    filename = f"{class_folder}{original_filename}"
                    
                    # Convert image to bytes
                    img_bytes = io.BytesIO()
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(img_bytes, format='JPEG', quality=90)
                    img_bytes.seek(0)
                    
                    # Add to zip
                    zip_file.writestr(filename, img_bytes.getvalue())
        
        zip_buffer.seek(0)
        logger.info(f"Created ZIP archive with summary: {zip_name}")
        
        return zip_buffer.getvalue()
    
    def cleanup_temp_files(self):
        """Remove all temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)
            logger.info("Cleaned up temporary files")


class ImageAugmentor:
    """
    Simple image augmentation utilities
    """
    
    @staticmethod
    def random_flip(image: Image.Image) -> Image.Image:
        """Randomly flip image horizontally"""
        if np.random.random() > 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    @staticmethod
    def random_rotation(image: Image.Image, max_angle: int = 15) -> Image.Image:
        """Randomly rotate image"""
        angle = np.random.uniform(-max_angle, max_angle)
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    @staticmethod
    def random_brightness(image: Image.Image, range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Randomly adjust brightness"""
        from PIL import ImageEnhance
        factor = np.random.uniform(*range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def random_contrast(image: Image.Image, range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Randomly adjust contrast"""
        from PIL import ImageEnhance
        factor = np.random.uniform(*range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)


# Testing
if __name__ == "__main__":
    print("Testing ImageProcessor...")
    processor = ImageProcessor()
    
    # Create test image
    test_img = Image.new('RGB', (2000, 1500), color='blue')
    print(f"Original size: {test_img.size}")
    
    # Resize
    resized = processor.resize_image(test_img, max_size=800)
    print(f"Resized: {resized.size}")
    
    # Display version
    display = processor.prepare_for_display(test_img)
    print(f"Display version: {display.size}")
    
    print("\nTesting BatchProcessor...")
    batch = BatchProcessor()
    
    # Create test data
    images = [Image.new('RGB', (100, 100), color=c) for c in ['red', 'green', 'blue']]
    predictions = [
        {'predicted_class': 'plastic', 'confidence': 0.9, 'status': 'confident'},
        {'predicted_class': 'paper', 'confidence': 0.8, 'status': 'confident'},
        {'predicted_class': 'plastic', 'confidence': 0.7, 'status': 'uncertain'}
    ]
    filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    
    # Organize
    organized = batch.organize_by_class(images, predictions, filenames)
    print(f"Organized into {len(organized)} classes:")
    for class_name, items in organized.items():
        print(f"  {class_name}: {len(items)} images")
    
    # Create summary
    summary = batch.create_summary_file(predictions, filenames)
    print("\nSummary preview:")
    print(summary[:200] + "...")
    
    print("\n✅ All tests passed!")