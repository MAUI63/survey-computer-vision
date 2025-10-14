# FastAPI server for static files and custom API routes
import os
from io import BytesIO

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

app = FastAPI()

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# OK, have an endpoint that serves images but first it removes the exif orientation data:
@app.get("/images/{image_path:path}")
async def get_image(image_path: str):
    full_path = os.path.join(IMAGE_DIR, image_path)
    if not os.path.isfile(full_path):
        return JSONResponse(status_code=404, content={"message": f"Image not found at {full_path}"})
    # Get the camera:
    camera = image_path.split("/")[2]
    print(f"Serving image {image_path} from camera {camera}")
    # rotation = {
    #     "r09": 0,
    #     "l28": 0,
    #     "r28": 180,
    #     "l09": 180,
    # }[camera.lower()]

    with Image.open(full_path) as img:
        # Create new image with no exif metadata, otherwise browser uses it, which we don't want:
        new = Image.new(img.mode, img.size)
        new.paste(img)
        # Rotate if needed:
        # if rotation != 0:
        #     print(f"Rotating image from camera {camera} by {rotation} degrees")
        #     new = new.rotate(rotation, expand=True)

        io = BytesIO()
        new.save(io, format="JPEG")
        io.seek(0)
    return StreamingResponse(iterfile(io, chunk_size=100000), media_type="image/jpeg")


def iterfile(file_like, chunk_size=65536):
    while True:
        chunk = file_like.read(chunk_size)
        if not chunk:
            break
        yield chunk


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument("app_data_dir", type=str, help="Directory to serve files from")
    parser.add_argument("img_dir", type=str, help="Directory to serve images from")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    STATIC_DIR = os.path.abspath(args.app_data_dir)
    app.mount("/files", StaticFiles(directory=STATIC_DIR), name="files")

    IMAGE_DIR = os.path.abspath(args.img_dir)

    uvicorn.run(app, host=args.host, port=args.port)
