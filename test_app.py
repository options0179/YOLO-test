import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import app

TEST_IMAGE = Path(__file__).parent / "test-image" / "input" / "test.jpeg"


class DetectApiTests(unittest.TestCase):
    def test_detect_rejects_non_image_upload(self):
        client = TestClient(app)

        response = client.post(
            "/detect",
            files={"file": ("note.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)

    def test_detect_image_returns_downloadable_png(self):
        client = TestClient(app)

        with TEST_IMAGE.open("rb") as f:
            response = client.post(
                "/detect/image",
                files={"file": ("test.jpeg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertIn("attachment", response.headers["content-disposition"])


if __name__ == "__main__":
    unittest.main()
