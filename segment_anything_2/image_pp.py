# image_preprocessor.py
import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from segment_anything_2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything_2.sam2.modeling.sam2_base import SAM2Base


class ImagePreprocessor:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def get_largest_mask(self, anns):
        max_area = 0
        largest_mask = None

        # Find the largest mask
        for ann in anns:
            mask = ann['segmentation']
            area = np.sum(mask)

            if area > max_area:
                max_area = area
                largest_mask = mask

        return largest_mask

    def get_masked_image(self, anns, image_np):
        largest_mask = self.get_largest_mask(anns)
        
        if largest_mask is not None:
            # Find bounding box of the largest mask
            coords = np.column_stack(np.where(largest_mask))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop the image to the bounding box
            cropped_image = image_np[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = largest_mask[y_min:y_max+1, x_min:x_max+1]
            
            # Apply the mask to the cropped image
            cropped_image[~cropped_mask] = 0

            return cropped_image
        return None

    def process_image(self, img_url):
        try:
            # Load the image from the URL
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')  # Convert image to RGB
            image_np = np.array(image)  # Convert to NumPy array

            # Automatically generate masks for the current image
            anns = self.mask_generator.generate(image_np)
            
            # Get the largest masked object
            masked_image_np = self.get_masked_image(anns, image_np)

            return image_np, masked_image_np

        except Exception as e:
            print(f"Error processing image from URL {img_url}: {e}")
            return None, None