import uvicorn
import sys
sys.path.append('/root/riya/pipeline_sam_blur/segment_anything_2')
print(sys.path)
from fastapi import FastAPI
from segment_anything_2 import fastapi_image_processor 
#from blur_det.fastapi_blur_detector import router_blur

app = FastAPI()

# Include the routers
# app.include_router(router_sam)
app.include_router(fastapi_image_processor.router,tags=['sam_blur'])
#app.include_router(router_blur)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7520)
