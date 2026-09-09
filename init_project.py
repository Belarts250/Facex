"""Initialize FaceX and download checksum-verified upstream models on request."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = {
    "detector_yunet.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "embedder_arcface.onnx": (
        "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
        "f3a6bc281e72f88862f5748b53be3d76b3b48f8f1ab1f4a537941bdc4e1b01da"),
}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name, url, expected):
    target = ROOT / "models" / name
    if target.exists() and target.stat().st_size:
        if checksum(target) == expected:
            print(f"Verified existing {name}")
            return
        raise ValueError(f"{target} is a different model; move it aside before downloading")
    temporary = target.with_suffix(".onnx.part")
    print(f"Downloading {name} ...", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "FaceX/1.0"})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
        if checksum(temporary) != expected:
            raise ValueError(f"Checksum mismatch for {name}; download not installed")
        temporary.replace(target)
        print(f"Verified {name} ({target.stat().st_size:,} bytes)")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--download-models", action="store_true", help="Download YuNet and ArcFace (~249 MiB total)")
    args = cli.parse_args()
    for folder in ("models", "data/db"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if args.download_models:
        for name, (url, digest) in MODELS.items():
            download(name, url, digest)
    print("FaceX directories ready. Enroll with: python -m src.enroll --name YOUR_NAME")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Setup failed: {error}") from error
