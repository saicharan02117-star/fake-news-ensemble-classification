"""Vercel Python application for the website and classification API."""

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlparse

from ml_engine import analyze_text, evaluation_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


class handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_file(self, relative_path: str, content_type: str, include_body: bool = True) -> None:
        body = (PROJECT_ROOT / relative_path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        cache_control = "no-cache, must-revalidate" if relative_path == "index.html" else "no-store, max-age=0"
        self.send_header("Cache-Control", cache_control)
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _route_get(self, include_body: bool = True) -> None:
        path = urlparse(self.path).path
        if path in STATIC_FILES:
            relative_path, content_type = STATIC_FILES[path]
            self._write_file(relative_path, content_type, include_body)
            return

        if path in {"/api/analyze", "/api/health"}:
            payload = {
                "status": "ready",
                "project": "Fake News Detection Using Ensemble Text Classification",
                "metrics": evaluation_metrics(),
            }
            if include_body:
                self._write_json(200, payload)
            else:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            return

        self._write_json(404, {"ok": False, "error": "Page not found."})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        self._route_get()

    def do_HEAD(self) -> None:
        self._route_get(include_body=False)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self._write_json(404, {"ok": False, "error": "API endpoint not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50000:
                raise ValueError("Submit a valid article with no more than 20,000 characters.")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            article = str(data.get("text", "")).strip()
            self._write_json(200, {"ok": True, "result": analyze_text(article)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._write_json(400, {"ok": False, "error": str(exc)})
        except Exception:
            self._write_json(500, {"ok": False, "error": "The classifier could not process this article."})
