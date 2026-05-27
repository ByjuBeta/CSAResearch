"""Website for the image classifier."""

from __future__ import annotations

import base64
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from image_model import classify_image


BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000


class ImageRequestHandler(SimpleHTTPRequestHandler):
    """Serve the website and answer prediction requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.path = "/static/index.html"
        elif path == "/training-curves":
            self.path = "/model/training_curves.png"
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_data = payload.get("image", "")
            image = decode_data_url(image_data)
            result = classify_image(image)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def decode_data_url(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Missing image data.")
    _, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data)).convert("RGB")


def main():
    server = ThreadingHTTPServer((HOST, PORT), ImageRequestHandler)
    print(f"Image classifier website running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
