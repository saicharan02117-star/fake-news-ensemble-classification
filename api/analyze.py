"""Vercel Python Function for article classification."""

from http.server import BaseHTTPRequestHandler
import json

from ml_engine import analyze_text, evaluation_metrics


class handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        self._write_json(
            200,
            {
                "status": "ready",
                "project": "Fake News Detection Using Ensemble Text Classification",
                "metrics": evaluation_metrics(),
            },
        )

    def do_POST(self) -> None:
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

