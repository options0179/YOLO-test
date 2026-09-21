import unittest

from fastapi.testclient import TestClient

from app import app


class DetectApiTests(unittest.TestCase):
    def test_detect_rejects_non_image_upload(self):
        client = TestClient(app)

        response = client.post(
            "/detect",
            files={"file": ("note.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
