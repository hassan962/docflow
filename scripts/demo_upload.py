"""Upload a folder of documents to a running DocFlow API and wait for the pipeline.

    uv run python scripts/demo_upload.py data/synthetic --until ocr_done

Reads the API key from DOCFLOW_API_KEY (environment or .env). Prints one line per
document with its final status, never the key.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
TERMINAL = {"failed", "approved", "indexed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--until", default="ocr_done", help="status to wait for")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    load_dotenv()
    key = os.environ.get("DOCFLOW_API_KEY")
    if not key:
        raise SystemExit("DOCFLOW_API_KEY is not set (add it to .env)")

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in SUFFIXES)
    with httpx.Client(base_url=args.api, headers={"X-API-Key": key}, timeout=60) as client:
        ids: dict[str, str] = {}
        for path in files:
            r = client.post("/v1/documents", files={"file": (path.name, path.read_bytes())})
            r.raise_for_status()
            body = r.json()
            ids[body["id"]] = path.name
            note = "  (duplicate)" if body["duplicate"] else ""
            print(f"{r.status_code} {path.name:<22} {body['id']}{note}")

        deadline = time.monotonic() + args.timeout
        pending = set(ids)
        while pending and time.monotonic() < deadline:
            for doc_id in list(pending):
                status = client.get(f"/v1/documents/{doc_id}").json()["status"]
                if status == args.until or status in TERMINAL:
                    pending.discard(doc_id)
            if pending:
                time.sleep(2)

        print()
        for doc_id, name in ids.items():
            d = client.get(f"/v1/documents/{doc_id}").json()
            conf = [p["mean_ocr_conf"] for p in d["pages"] if p["mean_ocr_conf"] is not None]
            mean = f"{sum(conf) / len(conf):.2f}" if conf else "-"
            reason = f"  {d['failure_reason']}" if d["failure_reason"] else ""
            print(f"{name:<22} {d['status']:<10} pages={d['page_count']} ocr_conf={mean}{reason}")
        if pending:
            raise SystemExit(f"{len(pending)} document(s) did not reach {args.until!r} in time")


if __name__ == "__main__":
    main()
