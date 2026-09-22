import argparse
import json
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).parent / "test-image" / "output"


def save_detection(image_path: Path, base_url: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    with image_path.open("rb") as f:
        detections = requests.post(f"{base_url}/detect", files={"file": f}).json()
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps(detections, indent=2, ensure_ascii=False))

    with image_path.open("rb") as f:
        annotated = requests.post(f"{base_url}/detect/image", files={"file": f})
    (OUTPUT_DIR / f"{stem}.png").write_bytes(annotated.content)

    print(f"saved {OUTPUT_DIR / f'{stem}.json'} and {OUTPUT_DIR / f'{stem}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    save_detection(args.image, args.base_url)
