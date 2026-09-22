from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import cv2
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


MODEL_NAME = "yolo26s.pt"

app = FastAPI()


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    return YOLO(MODEL_NAME)


async def load_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported")

    try:
        return Image.open(BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def run_detection(image: Image.Image):
    return get_model()(image, imgsz=640, batch=1, verbose=False)[0]


def parse_detections(result) -> list[dict]:
    detections = []
    names = result.names

    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    ):
        class_id = int(class_id)
        detections.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": float(confidence),
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                },
            }
        )

    return detections


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = await load_image(file)
    result = run_detection(image)
    return {"detections": parse_detections(result)}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    image = await load_image(file)
    result = run_detection(image)

    annotated_bgr = result.plot()
    ok, png = cv2.imencode(".png", annotated_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")

    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=detected.png"},
    )
