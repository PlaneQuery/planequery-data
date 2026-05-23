#!/usr/bin/env python3

from pathlib import Path
from urllib.parse import quote
import time
import xml.etree.ElementTree as ET

import requests

BUCKET_URL = "https://s3.opensky-network.org/data-samples"
PREFIX = "trino-tables/state_vectors/"
OUT_DIR = Path("opensky_state_vectors")

NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix: str):
    token = None

    while True:
        url = (
            f"{BUCKET_URL}"
            f"?list-type=2"
            f"&prefix={quote(prefix, safe='/=')}"
        )

        if token:
            url += f"&continuation-token={quote(token, safe='/=')}"

        print(f"Listing: {url}")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        found = 0

        for obj in root.findall("s3:Contents", NS):
            key = obj.findtext("s3:Key", default="", namespaces=NS)

            if key.endswith(".parquet"):
                found += 1
                yield key

        print(f"Found {found} parquet files in page")

        is_truncated = root.findtext(
            "s3:IsTruncated",
            default="false",
            namespaces=NS,
        )

        if is_truncated != "true":
            break

        token = root.findtext(
            "s3:NextContinuationToken",
            namespaces=NS,
        )

        if not token:
            break


def download_key(key: str):
    url = f"{BUCKET_URL}/{quote(key, safe='/=')}"

    out_path = OUT_DIR / key
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Skipping existing: {out_path}")
        return

    for attempt in range(1, 11):
        try:
            print(f"Downloading attempt {attempt}: {key}")

            with requests.get(
                url,
                stream=True,
                timeout=(30, 300),
            ) as response:

                response.raise_for_status()

                total = 0

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(
                        chunk_size=8 * 1024 * 1024
                    ):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)

                            print(
                                f"\r{key} "
                                f"{total / 1024 / 1024:.1f} MB",
                                end="",
                                flush=True,
                            )

                print()

            tmp_path.rename(out_path)

            print(f"Finished: {out_path}")
            return

        except KeyboardInterrupt:
            print("\nInterrupted")

            if tmp_path.exists():
                tmp_path.unlink()

            raise

        except requests.RequestException as e:
            print()
            print(f"Download failed: {e}")

            if tmp_path.exists():
                tmp_path.unlink()

            if attempt < 10:
                sleep_seconds = min(60, 5 * attempt)

                print(f"Retrying in {sleep_seconds}s...")

                time.sleep(sleep_seconds)
                continue

            print(f"Giving up on: {key}")
            return


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    keys = list(list_objects(PREFIX))

    print()
    print(f"Total parquet files found: {len(keys)}")
    print()

    for i, key in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}]")
        download_key(key)
        print()


if __name__ == "__main__":
    main()