import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SEBS_ROOT = Path(os.environ.get("SEBS_ROOT", "/sebs"))
BENCHMARK_ROOT = SEBS_ROOT / "benchmarks"
LOCAL_WORKLOAD_ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET_SECONDS = 200.0
DEFAULT_TARGET_ITERATIONS = 0
DEFAULT_INPUT_SIZE = "large"
RUNTIME_PARENT = Path("/tmp/sebs_runtime_packages")
DATA_PARENT = Path("/tmp/sebs_local_data")
STORAGE_PARENT = Path("/tmp/sebs_local_storage")
NOSQL_PARENT = Path("/tmp/sebs_local_nosql")

LOCAL_STORAGE_SHIM = """\
from io import BytesIO
import os
from pathlib import Path
import shutil
import uuid


class storage:
    instance = None

    def __init__(self):
        self.root = Path(os.environ["SEBS_LOCAL_STORAGE_ROOT"])

    @staticmethod
    def unique_name(name):
        base, extension = os.path.splitext(name)
        return f"{base}.{str(uuid.uuid4()).split('-')[0]}{extension}"

    def _object_path(self, bucket, key):
        return self.root / bucket / key

    def upload(self, bucket, file, filepath):
        key_name = storage.unique_name(file)
        destination = self._object_path(bucket, key_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filepath, destination)
        return key_name

    def download(self, bucket, file, filepath):
        source = self._object_path(bucket, file)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, filepath)

    def download_directory(self, bucket, prefix, path):
        source_root = self._object_path(bucket, prefix)
        destination_root = Path(path)
        for source in source_root.rglob("*"):
            if source.is_dir():
                continue
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def upload_stream(self, bucket, file, bytes_data):
        key_name = storage.unique_name(file)
        destination = self._object_path(bucket, key_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as f:
            f.write(bytes_data.read())
        return key_name

    def download_stream(self, bucket, file):
        source = self._object_path(bucket, file)
        with open(source, "rb") as f:
            return f.read()

    @staticmethod
    def get_instance():
        if storage.instance is None:
            storage.instance = storage()
        return storage.instance
"""

LOCAL_NOSQL_SHIM = """\
import json
import os
from pathlib import Path


class nosql:
    instance = None

    def __init__(self):
        self.root = Path(os.environ["SEBS_LOCAL_NOSQL_ROOT"])
        self.root.mkdir(parents=True, exist_ok=True)

    def _table_path(self, table_name):
        return self.root / f"{table_name}.json"

    def _load_table(self, table_name):
        table_path = self._table_path(table_name)
        if not table_path.exists():
            return {}
        with open(table_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_table(self, table_name, data):
        table_path = self._table_path(table_name)
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _key(self, primary_key, secondary_key):
        payload = {
            primary_key[0]: primary_key[1],
            secondary_key[0]: secondary_key[1],
        }
        return json.dumps(payload, sort_keys=True)

    def insert(self, table_name, primary_key, secondary_key, data):
        table = self._load_table(table_name)
        item = dict(data)
        item[primary_key[0]] = primary_key[1]
        item[secondary_key[0]] = secondary_key[1]
        table[self._key(primary_key, secondary_key)] = item
        self._save_table(table_name, table)

    def get(self, table_name, primary_key, secondary_key):
        table = self._load_table(table_name)
        key = self._key(primary_key, secondary_key)
        item = table.get(key)
        if item is None:
            raise KeyError(f"Missing item for table={table_name}, key={key}")
        return item

    def update(self, table_name, primary_key, secondary_key, updates):
        table = self._load_table(table_name)
        key = self._key(primary_key, secondary_key)
        item = dict(table.get(key, {}))
        item.update(updates)
        item[primary_key[0]] = primary_key[1]
        item[secondary_key[0]] = secondary_key[1]
        table[key] = item
        self._save_table(table_name, table)

    def query(self, table_name, primary_key, _):
        table = self._load_table(table_name)
        return [
            item for item in table.values()
            if item.get(primary_key[0]) == primary_key[1]
        ]

    def delete(self, table_name, primary_key, secondary_key):
        table = self._load_table(table_name)
        key = self._key(primary_key, secondary_key)
        if key in table:
            del table[key]
            self._save_table(table_name, table)

    @staticmethod
    def get_instance():
        if nosql.instance is None:
            nosql.instance = nosql()
        return nosql.instance
"""


SEBS_WORKLOADS = {
    "sebs_dynamic_html_known": {
        "display_name": "Web (dynamic-html)",
        "suite": "SeBS",
        "workload_type": "web",
        "partition": "known",
        "benchmark_name": "dynamic-html",
        "benchmark_id": "110.dynamic-html",
        "relative_dir": Path("100.webapps") / "110.dynamic-html",
        "input_size": "large",
    },
    "sebs_graph_bfs_known": {
        "display_name": "Scientific (graph-bfs)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "known",
        "benchmark_name": "graph-bfs",
        "benchmark_id": "503.graph-bfs",
        "relative_dir": Path("500.scientific") / "503.graph-bfs",
        "input_size": "large",
    },
    "sebs_graph_mst_known": {
        "display_name": "Scientific (graph-mst)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "known",
        "benchmark_name": "graph-mst",
        "benchmark_id": "502.graph-mst",
        "relative_dir": Path("500.scientific") / "502.graph-mst",
        "input_size": "large",
    },
    "sebs_uploader_known": {
        "display_name": "Web (uploader)",
        "suite": "SeBS",
        "workload_type": "web",
        "partition": "known",
        "benchmark_name": "uploader",
        "benchmark_id": "120.uploader",
        "relative_dir": Path("100.webapps") / "120.uploader",
        "input_size": "large",
        "needs_storage": True,
    },
    "sebs_video_processing_known": {
        "display_name": "Multimedia (video-processing)",
        "suite": "SeBS",
        "workload_type": "multimedia",
        "partition": "known",
        "benchmark_name": "video-processing",
        "benchmark_id": "220.video-processing",
        "relative_dir": Path("200.multimedia") / "220.video-processing",
        "input_size": "large",
        "needs_storage": True,
        "copy_resources": ["resources"],
        "system_ffmpeg_shim": True,
        "event_duration_seconds": 2,
    },
    "sebs_dna_visualisation_known": {
        "display_name": "Scientific (dna-visualisation)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "known",
        "benchmark_name": "dna-visualisation",
        "benchmark_id": "504.dna-visualisation",
        "relative_dir": Path("500.scientific") / "504.dna-visualisation",
        "input_size": "small",
        "needs_storage": True,
    },
    "sebs_compression_known": {
        "display_name": "Utility (compression)",
        "suite": "SeBS",
        "workload_type": "utility",
        "partition": "known",
        "benchmark_name": "compression",
        "benchmark_id": "311.compression",
        "relative_dir": Path("300.utilities") / "311.compression",
        "input_size": "large",
        "needs_storage": True,
    },
    "sebs_crud_api_known": {
        "display_name": "Web (crud-api)",
        "suite": "SeBS",
        "workload_type": "web",
        "partition": "known",
        "benchmark_name": "crud-api",
        "benchmark_id": "130.crud-api",
        "relative_dir": Path("100.webapps") / "130.crud-api",
        "input_size": "large",
        "needs_nosql": True,
    },
    "sebs_compression_unseen": {
        "display_name": "Utility (compression)",
        "suite": "SeBS",
        "workload_type": "utility",
        "partition": "unseen",
        "benchmark_name": "compression",
        "benchmark_id": "311.compression",
        "relative_dir": Path("300.utilities") / "311.compression",
        "input_size": "large",
        "needs_storage": True,
    },
    "sebs_graph_mst_unseen": {
        "display_name": "Scientific (graph-mst)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "unseen",
        "benchmark_name": "graph-mst",
        "benchmark_id": "502.graph-mst",
        "relative_dir": Path("500.scientific") / "502.graph-mst",
        "input_size": "large",
    },
    "sebs_uploader_unseen": {
        "display_name": "Web (uploader)",
        "suite": "SeBS",
        "workload_type": "web",
        "partition": "unseen",
        "benchmark_name": "uploader",
        "benchmark_id": "120.uploader",
        "relative_dir": Path("100.webapps") / "120.uploader",
        "input_size": "large",
        "needs_storage": True,
    },
    "sebs_video_processing_unseen": {
        "display_name": "Multimedia (video-processing)",
        "suite": "SeBS",
        "workload_type": "multimedia",
        "partition": "unseen",
        "benchmark_name": "video-processing",
        "benchmark_id": "220.video-processing",
        "relative_dir": Path("200.multimedia") / "220.video-processing",
        "input_size": "large",
        "needs_storage": True,
        "copy_resources": ["resources"],
        "system_ffmpeg_shim": True,
        "event_duration_seconds": 2,
    },
    "sebs_dna_visualisation_unseen": {
        "display_name": "Scientific (dna-visualisation)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "unseen",
        "benchmark_name": "dna-visualisation",
        "benchmark_id": "504.dna-visualisation",
        "relative_dir": Path("500.scientific") / "504.dna-visualisation",
        "input_size": "small",
        "needs_storage": True,
    },
    "functionbench_download_upload_known": {
        "display_name": "FunctionBench (download-upload)",
        "suite": "FunctionBench",
        "workload_type": "web",
        "partition": "known",
        "benchmark_name": "download-upload",
        "benchmark_id": "FB.download-upload",
        "local_relative_dir": Path("functionbench_download_upload"),
        "input_size": "large",
        "needs_storage": True,
        "data_generator": "functionbench_download_upload",
    },
    "functionbench_download_upload_unseen": {
        "display_name": "FunctionBench (download-upload)",
        "suite": "FunctionBench",
        "workload_type": "web",
        "partition": "unseen",
        "benchmark_name": "download-upload",
        "benchmark_id": "FB.download-upload",
        "local_relative_dir": Path("functionbench_download_upload"),
        "input_size": "large",
        "needs_storage": True,
        "data_generator": "functionbench_download_upload",
    },
}

RESOURCE_PROFILE_MAP = {
    "sebs_compression_known": ("cpu", 1),
    "sebs_compression_unseen": ("cpu", 1),
    "sebs_video_processing_known": ("cpu", 2),
    "sebs_video_processing_unseen": ("cpu", 2),
    "sebs_graph_bfs_known": ("memory", 2),
    "sebs_graph_mst_known": ("memory", 3),
    "sebs_graph_mst_unseen": ("memory", 3),
    "sebs_dynamic_html_known": ("mixed", 2),
    "sebs_crud_api_known": ("mixed", 4),
    "sebs_uploader_known": ("mixed", 5),
    "sebs_uploader_unseen": ("mixed", 5),
    "sebs_dna_visualisation_known": ("mixed", 6),
    "sebs_dna_visualisation_unseen": ("mixed", 6),
    "functionbench_download_upload_known": ("mixed", 7),
    "functionbench_download_upload_unseen": ("mixed", 7),
}

RESOURCE_PROFILE_PREFIX = {
    "cpu": "cpu",
    "memory": "mem",
    "mixed": "mix",
}

for workload_key, (resource_profile, resource_profile_index) in RESOURCE_PROFILE_MAP.items():
    spec = SEBS_WORKLOADS[workload_key]
    profile_prefix = RESOURCE_PROFILE_PREFIX[resource_profile]
    resource_profile_label = f"{profile_prefix}{resource_profile_index}"
    spec["resource_profile"] = resource_profile
    spec["resource_profile_index"] = resource_profile_index
    spec["resource_profile_label"] = resource_profile_label
    spec["plot_workload_name"] = f"{spec['benchmark_name']}_{resource_profile_label}"

KNOWN_SEBS_WORKLOADS = [
    "sebs_compression_known",
    "sebs_video_processing_known",
    "sebs_graph_bfs_known",
    "sebs_graph_mst_known",
    "sebs_dynamic_html_known",
    "sebs_crud_api_known",
    "sebs_uploader_known",
    "sebs_dna_visualisation_known",
]

UNSEEN_SEBS_WORKLOADS = [
    "sebs_graph_mst_unseen",
    "sebs_uploader_unseen",
    "sebs_video_processing_unseen",
    "functionbench_download_upload_unseen",
]


def load_event(argv):
    if len(argv) <= 1:
        return {}

    raw = argv[1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_arg": raw}


def benchmark_dir(spec):
    if "local_relative_dir" in spec:
        return LOCAL_WORKLOAD_ROOT / spec["local_relative_dir"]
    return BENCHMARK_ROOT / spec["relative_dir"]


def python_source_dir(spec):
    return benchmark_dir(spec) / "python"


def input_script_path(spec):
    return benchmark_dir(spec) / "input.py"


def ensure_sebs_exists():
    if not BENCHMARK_ROOT.exists():
        raise FileNotFoundError(
            f"SeBS benchmark root not found at {BENCHMARK_ROOT}. "
            "Clone the official SeBS repository and mount it into the container at /sebs."
        )


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_runtime_package(workload_key, spec):
    source_dir = python_source_dir(spec)
    package_name = f"codex_sebs_{workload_key}"
    package_dir = RUNTIME_PARENT / package_name

    if package_dir.exists():
        shutil.rmtree(package_dir)
    shutil.copytree(source_dir, package_dir)

    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    for relative_name in spec.get("copy_resources", []):
        source_path = benchmark_dir(spec) / relative_name
        destination_path = package_dir / relative_name
        if destination_path.exists():
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        elif source_path.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

    if spec.get("needs_storage"):
        (package_dir / "storage.py").write_text(LOCAL_STORAGE_SHIM, encoding="utf-8")
    if spec.get("needs_nosql"):
        (package_dir / "nosql.py").write_text(LOCAL_NOSQL_SHIM, encoding="utf-8")

    if spec.get("system_ffmpeg_shim"):
        ffmpeg_dir = package_dir / "ffmpeg"
        ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_shim = ffmpeg_dir / "ffmpeg"
        ffmpeg_shim.write_text("#!/bin/sh\nexec /usr/bin/ffmpeg \"$@\"\n", encoding="utf-8")
        ffmpeg_shim.chmod(0o755)

    parent_str = str(RUNTIME_PARENT)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)

    importlib.invalidate_caches()
    for mod_name in list(sys.modules.keys()):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    return importlib.import_module(f"{package_name}.function")


def ensure_data_dir(workload_key, spec):
    data_dir = DATA_PARENT / workload_key
    data_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = spec["benchmark_name"]
    if benchmark_name == "thumbnailer":
        ensure_thumbnailer_data(data_dir)
    elif benchmark_name == "compression":
        ensure_compression_data(data_dir)
    elif benchmark_name == "image-recognition":
        ensure_image_recognition_data(data_dir)
    elif benchmark_name == "uploader":
        ensure_uploader_data(data_dir)
    elif benchmark_name == "video-processing":
        ensure_video_processing_data(data_dir)
    elif benchmark_name == "dna-visualisation":
        ensure_dna_visualisation_data(data_dir, spec.get("input_size", DEFAULT_INPUT_SIZE))
    elif spec.get("data_generator") == "functionbench_download_upload":
        ensure_functionbench_download_upload_data(data_dir)

    return data_dir


def ensure_thumbnailer_data(data_dir):
    from PIL import Image, ImageDraw

    image_path = data_dir / "6_astronomy-desktop-wallpaper-evening-1624438.jpg"
    if image_path.exists():
        return

    image = Image.new("RGB", (3840, 2160), color=(8, 12, 30))
    draw = ImageDraw.Draw(image)
    for idx in range(0, 3840, 24):
        color = ((idx * 17) % 255, (idx * 31) % 255, (idx * 7) % 255)
        draw.line((idx, 0, 3840 - idx, 2160), fill=color, width=8)
    image.save(image_path, format="JPEG", quality=95)


def ensure_compression_data(data_dir):
    dataset_dir = data_dir / "dataset-large"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    payload = ("serverless-benchmark-data\n" * 4096).encode("utf-8")
    for idx in range(4):
        file_path = dataset_dir / f"payload_{idx:02d}.txt"
        file_path.write_bytes(payload)


def ensure_image_recognition_data(data_dir):
    from PIL import Image, ImageDraw

    model_dir = data_dir / "model"
    image_dir = data_dir / "fake-resnet"
    model_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "sample.jpg"
    if not image_path.exists():
        image = Image.new("RGB", (256, 256), color=(22, 38, 74))
        draw = ImageDraw.Draw(image)
        for offset in range(0, 256, 16):
            color = ((offset * 13) % 255, (offset * 29) % 255, (offset * 47) % 255)
            draw.rectangle((offset, 0, min(offset + 8, 255), 255), fill=color)
        draw.ellipse((48, 48, 208, 208), outline=(255, 255, 255), width=6)
        image.save(image_path, format="JPEG", quality=92)

    val_map_path = image_dir / "val_map.txt"
    val_map_path.write_text("sample.jpg 0\n", encoding="utf-8")

    model_path = model_dir / "resnet50-19c8e357.pth"
    if not model_path.exists():
        try:
            import torch
            from torchvision.models import resnet50
        except Exception as exc:
            raise RuntimeError(
                "sebs_image_recognition_unseen requires torch and torchvision in the runner image."
            ) from exc

        try:
            model = resnet50(weights=None)
        except TypeError:
            model = resnet50(pretrained=False)
        torch.save(model.state_dict(), model_path)

    cpp_model_path = model_dir / "resnet50.pt"
    if not cpp_model_path.exists():
        cpp_model_path.write_bytes(b"placeholder-for-local-python-runner\n")


def ensure_uploader_data(data_dir):
    file_path = data_dir / "upload-large.bin"
    if file_path.exists() and file_path.stat().st_size >= (64 * 1024 * 1024):
        return

    chunk = (b"serverless-uploader-payload-" * 1024)
    target_size = 64 * 1024 * 1024
    written = 0
    with open(file_path, "wb") as f:
        while written < target_size:
            remaining = target_size - written
            block = chunk if remaining >= len(chunk) else chunk[:remaining]
            f.write(block)
            written += len(block)


def ensure_video_processing_data(data_dir):
    video_path = data_dir / "sample-large.mp4"
    if video_path.exists() and video_path.stat().st_size > 0:
        return

    profiles = [
        {"size": "640x360", "rate": "15", "seconds": "4"},
        {"size": "480x270", "rate": "12", "seconds": "3"},
        {"size": "320x180", "rate": "10", "seconds": "2"},
    ]

    last_error = None
    for profile in profiles:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={profile['size']}:rate={profile['rate']}",
            "-t",
            profile["seconds"],
            "-c:v",
            "mpeg4",
            "-q:v",
            "6",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if video_path.exists() and video_path.stat().st_size > 0:
                return
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for sebs_video_processing_known. Rebuild the runner image after installing ffmpeg.") from exc
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if video_path.exists():
                video_path.unlink()

    raise RuntimeError("Unable to generate sample video for sebs_video_processing_known under the current memory cap.") from last_error


def ensure_dna_visualisation_data(data_dir, input_size):
    fasta_path = data_dir / "sequence-large.fasta"
    sequence = "ACGTGGTCTTAA"

    profiles = {
        "test": {"lines_per_record": 128, "records": 4},
        "small": {"lines_per_record": 384, "records": 6},
        "large": {"lines_per_record": 1024, "records": 12},
    }
    profile = profiles.get(str(input_size), profiles["small"])

    with open(fasta_path, "w", encoding="utf-8") as f:
        for record_idx in range(profile["records"]):
            f.write(f">sequence_{record_idx:02d}\n")
            for _ in range(profile["lines_per_record"]):
                f.write(sequence * 10)
                f.write("\n")


def ensure_functionbench_download_upload_data(data_dir):
    profiles = {
        "payload-test.bin": 2 * 1024 * 1024,
        "payload-small.bin": 16 * 1024 * 1024,
        "payload-large.bin": 64 * 1024 * 1024,
    }

    chunk = (b"functionbench-download-upload-payload-" * 1024)
    for filename, target_size in profiles.items():
        path = data_dir / filename
        if path.exists() and path.stat().st_size >= target_size:
            continue
        written = 0
        with open(path, "wb") as f:
            while written < target_size:
                remaining = target_size - written
                block = chunk if remaining >= len(chunk) else chunk[:remaining]
                f.write(block)
                written += len(block)


def copy_to_storage(storage_root, bucket, key, filepath):
    destination = storage_root / bucket / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(filepath, destination)


def clear_output_paths(storage_root, bucket, output_paths):
    for output_path in output_paths:
        path = storage_root / bucket / output_path
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def build_event(workload_key, spec, input_size):
    data_dir = ensure_data_dir(workload_key, spec)
    input_module = load_module_from_path(
        f"codex_sebs_input_{workload_key}",
        input_script_path(spec),
    )

    storage_root = STORAGE_PARENT / workload_key
    if storage_root.exists():
        shutil.rmtree(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    benchmark_bucket = f"codex-{workload_key}"
    buckets_count_fn = getattr(input_module, "buckets_count", None)
    if callable(buckets_count_fn):
        input_count, output_count = buckets_count_fn()
    else:
        input_count, output_count = (0, 0)

    input_paths = [
        f"{spec['benchmark_name'].replace('.', '_')}/input/{idx}"
        for idx in range(input_count)
    ]
    output_paths = [
        f"{spec['benchmark_name'].replace('.', '_')}/output/{idx}"
        for idx in range(output_count)
    ]

    def upload_func(bucket_idx, key, filepath):
        copy_to_storage(storage_root, benchmark_bucket, f"{input_paths[bucket_idx]}/{key}", filepath)

    nosql_root = NOSQL_PARENT / workload_key
    if nosql_root.exists():
        shutil.rmtree(nosql_root)
    nosql_root.mkdir(parents=True, exist_ok=True)
    os.environ["SEBS_LOCAL_NOSQL_ROOT"] = str(nosql_root)

    def nosql_upload(_benchmark_name, table_name, data, primary_key, secondary_key):
        if not spec.get("needs_nosql"):
            raise RuntimeError("SeBS NoSQL workloads are not enabled in this local wrapper.")

        table_path = nosql_root / f"{table_name}.json"
        if table_path.exists():
            with open(table_path, "r", encoding="utf-8") as f:
                table = json.load(f)
        else:
            table = {}

        item = dict(data)
        item[primary_key[0]] = primary_key[1]
        item[secondary_key[0]] = secondary_key[1]
        key = json.dumps(
            {
                primary_key[0]: primary_key[1],
                secondary_key[0]: secondary_key[1],
            },
            sort_keys=True,
        )
        table[key] = item
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(table, f)

    event = input_module.generate_input(
        str(data_dir),
        input_size,
        benchmark_bucket,
        input_paths,
        output_paths,
        upload_func,
        nosql_upload,
    )

    if spec["benchmark_name"] == "uploader":
        local_upload_file = (data_dir / "upload-large.bin").resolve()
        event["object"]["url"] = local_upload_file.as_uri()
    elif spec["benchmark_name"] == "video-processing":
        event["object"]["duration"] = int(spec.get("event_duration_seconds", event["object"].get("duration", 1)))

    return event, storage_root, benchmark_bucket, output_paths


def run_registered_workload(workload_key, event=None):
    if workload_key not in SEBS_WORKLOADS:
        raise KeyError(f"Unknown SeBS workload: {workload_key}")

    spec = SEBS_WORKLOADS[workload_key]
    if "local_relative_dir" not in spec:
        ensure_sebs_exists()
    event = event or {}
    target_seconds = float(event.get("target_seconds", DEFAULT_TARGET_SECONDS))
    target_iterations_raw = event.get("target_iterations", DEFAULT_TARGET_ITERATIONS)
    try:
        target_iterations = int(target_iterations_raw) if target_iterations_raw is not None else 0
    except (TypeError, ValueError):
        target_iterations = 0
    target_iterations = max(target_iterations, 0)
    input_size = str(event.get("input_size", spec.get("input_size", DEFAULT_INPUT_SIZE)))

    if spec.get("needs_storage"):
        generated_event, storage_root, bucket, output_paths = build_event(workload_key, spec, input_size)
        os.environ["SEBS_LOCAL_STORAGE_ROOT"] = str(storage_root)
    else:
        generated_event, storage_root, bucket, output_paths = build_event(workload_key, spec, input_size)

    function_module = ensure_runtime_package(workload_key, spec)
    handler = getattr(function_module, "handler")

    work_mode = "fixed_work" if target_iterations > 0 else "fixed_duration"
    if work_mode == "fixed_work":
        print(
            f"Running {spec['display_name']} "
            f"[{spec['suite']}, {spec['partition']}, input {input_size}] "
            f"for {target_iterations} iterations...",
            file=sys.stderr,
        )
    else:
        print(
            f"Running {spec['display_name']} "
            f"[{spec['suite']}, {spec['partition']}, input {input_size}] "
            f"for at least {target_seconds:.0f}s...",
            file=sys.stderr,
        )

    start_t = time.time()
    iterations = 0

    while True:
        if work_mode == "fixed_work":
            if iterations >= target_iterations:
                break
        elif (time.time() - start_t) >= target_seconds:
            break
        handler(generated_event)
        if output_paths:
            clear_output_paths(storage_root, bucket, output_paths)
        iterations += 1

    elapsed = time.time() - start_t
    summary = {
        "workload_key": workload_key,
        "display_name": spec["display_name"],
        "suite": spec["suite"],
        "workload_type": spec["workload_type"],
        "partition": spec["partition"],
        "benchmark_name": spec["benchmark_name"],
        "resource_profile": spec.get("resource_profile"),
        "resource_profile_index": spec.get("resource_profile_index"),
        "resource_profile_label": spec.get("resource_profile_label"),
        "plot_workload_name": spec.get("plot_workload_name"),
        "input_profile": input_size,
        "benchmark_id": spec["benchmark_id"],
        "target_seconds": target_seconds,
        "target_iterations": target_iterations if target_iterations > 0 else None,
        "work_mode": work_mode,
        "elapsed_seconds": elapsed,
        "iterations_completed": iterations,
        "command_runs": iterations,
    }
    print(json.dumps(summary))
    return summary


def main(workload_key):
    event = load_event(sys.argv)
    run_registered_workload(workload_key, event)
