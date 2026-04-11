import hashlib
import os
from pathlib import Path

from . import storage


client = storage.storage.get_instance()


def handler(event):
    bucket = event.get("bucket", {}).get("bucket")
    input_prefix = event.get("bucket", {}).get("input")
    output_prefix = event.get("bucket", {}).get("output")
    key = event.get("object", {}).get("key")

    if not bucket or not input_prefix or not output_prefix or not key:
        raise ValueError("Invalid event for FunctionBench download-upload workload.")

    source_key = os.path.join(input_prefix, key)
    download_path = Path("/tmp") / key
    upload_path = Path("/tmp") / f"processed-{key}"

    client.download(bucket, source_key, str(download_path))

    hasher = hashlib.sha256()
    with open(download_path, "rb") as src, open(upload_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            dst.write(chunk)
        dst.write(hasher.hexdigest().encode("ascii"))

    uploaded_key = client.upload(bucket, os.path.join(output_prefix, key), str(upload_path))

    return {
        "result": {
            "bucket": bucket,
            "input_key": source_key,
            "output_key": uploaded_key,
            "digest": hasher.hexdigest(),
        }
    }
