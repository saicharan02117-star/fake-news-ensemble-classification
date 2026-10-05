"""Small local server for the static site and Python analysis API."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from ml_engine import analyze_text, evaluation_metrics


ROOT = Path(__file__).resolve().parent


class DevHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/api/analyze":
            return self._json(200, {"status": "ready", "metrics": evaluation_metrics()})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._json(404, {"ok": False, "error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            return self._json(200, {"ok": True, "result": analyze_text(str(payload.get("text", "")))})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"ok": False, "error": str(exc)})

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DevHandler)
    print("Open http://127.0.0.1:8000")
    server.serve_forever()

