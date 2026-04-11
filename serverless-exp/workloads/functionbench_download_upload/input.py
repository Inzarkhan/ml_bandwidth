from pathlib import Path


FILE_BY_SIZE = {
    "test": "payload-test.bin",
    "small": "payload-small.bin",
    "large": "payload-large.bin",
}


def buckets_count():
    return (1, 1)


def generate_input(data_dir, size, benchmarks_bucket, input_buckets, output_buckets, upload_func, nosql_func):
    filename = FILE_BY_SIZE.get(size, FILE_BY_SIZE["large"])
    source_path = Path(data_dir) / filename
    upload_func(0, filename, str(source_path))
    return {
        "bucket": {
            "bucket": benchmarks_bucket,
            "input": input_buckets[0],
            "output": output_buckets[0],
        },
        "object": {
            "key": filename,
        },
    }
