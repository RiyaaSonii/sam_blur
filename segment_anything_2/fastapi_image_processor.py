from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from segment_anything_2.image_pp import ImagePreprocessor
from segment_anything_2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything_2.sam2.modeling.sam2_base import SAM2Base
from PIL import Image
import io
import base64

# Initialize the router
router_sam = APIRouter()

# Initialize SAM2AutomaticMaskGenerator and ImagePreprocessor
mask_generator = SAM2AutomaticMaskGenerator()
image_preprocessor = ImagePreprocessor(mask_generator)

# Define the request and response models
class ImageRequest(BaseModel):
    img_url: str

class ImageResponse(BaseModel):
    original_image: str
    masked_image: str

@router_sam.post("/process-image/", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    try:
        # Process the image using the ImagePreprocessor class
        original_image_np, masked_image_np = image_preprocessor.process_image(request.img_url)

        if original_image_np is None or masked_image_np is None:
            raise HTTPException(status_code=500, detail="Failed to process the image")

        # Convert images to base64 strings
        def image_to_base64(image_np):
            buffered = io.BytesIO()
            img = Image.fromarray(image_np)
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        original_image_base64 = image_to_base64(original_image_np)
        masked_image_base64 = image_to_base64(masked_image_np)

        return ImageResponse(
            original_image=original_image_base64,
            masked_image=masked_image_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
