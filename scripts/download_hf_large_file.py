#!/usr/bin/env python3
"""Download a large public Hugging Face file with verified HTTP ranges."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=8_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=12)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.truncate(args.size)

    chunks = [
        (start, min(start + args.chunk_size, args.size) - 1)
        for start in range(0, args.size, args.chunk_size)
    ]

    def fetch(chunk: tuple[int, int]) -> tuple[int, int]:
        start, end = chunk
        expected = end - start + 1
        for attempt in range(1, args.retries + 1):
            try:
                response = requests.get(
                    args.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(20, 90),
                )
                response.raise_for_status()
                payload = response.content
                content_range = response.headers.get("content-range", "")
                if response.status_code != 206 or len(payload) != expected or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(
                        f"bad range response status={response.status_code} bytes={len(payload)} content-range={content_range!r}"
                    )
                fd = os.open(args.output, os.O_WRONLY)
                try:
                    os.pwrite(fd, payload, start)
                finally:
                    os.close(fd)
                return chunk
            except Exception as error:
                if attempt == args.retries:
                    raise RuntimeError(f"range {start}-{end} failed after {attempt} attempts: {error}") from error
                time.sleep(min(10, attempt))
        raise AssertionError("unreachable")

    done_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, chunk) for chunk in chunks]
        for index, future in enumerate(as_completed(futures), 1):
            start, end = future.result()
            done_bytes += end - start + 1
            print(f"chunks={index}/{len(chunks)} bytes={done_bytes}/{args.size}", flush=True)

    actual = args.output.stat().st_size
    if actual != args.size:
        raise RuntimeError(f"size mismatch: {actual} != {args.size}")
    print(f"complete: {args.output} ({actual} bytes)")


if __name__ == "__main__":
    main()
