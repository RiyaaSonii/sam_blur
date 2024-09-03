from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from blur_det.blur_detection import BlurDetector
from PIL import Image
import numpy as np
import io
import base64

# Initialize the router
router_blur = APIRouter()

# Initialize BlurDetector
blur_detector = BlurDetector()

# Define the request and response models
class BlurRequest(BaseModel):
    image_base64: str

class BlurResponse(BaseModel):
    blur_score: float
    is_blurry: bool

@router_blur.post("/detect-blur/", response_model=BlurResponse)
async def detect_blur(request: BlurRequest):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
        image_np = np.array(Image.open(io.BytesIO(image_data)))

        # Compute blur score
        blur_score, is_blurry = blur_detector.compute_blur_score(image_np)

        return BlurResponse(
            blur_score=blur_score,
            is_blurry=is_blurry
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
