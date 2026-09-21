from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


MODEL_NAME = "yolo26s.pt"

app = FastAPI()


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    return YOLO(MODEL_NAME)


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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported")

    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    result = get_model()(image, imgsz=640, batch=1, verbose=False)[0]
    return {"detections": parse_detections(result)}
