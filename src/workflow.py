"""Shared command-line arguments and webcam lifetime management."""
import argparse
from contextlib import contextmanager
import cv2
from .config import DATABASE, DETECTOR, EMBEDDER
from .embed import ArcFace
from .landmarks import FaceDetector


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--camera", type=int, default=0)
    result.add_argument("--detector", default=str(DETECTOR))
    result.add_argument("--model", default=str(EMBEDDER))
    result.add_argument("--db", default=str(DATABASE))
    result.add_argument("--preprocessing", choices=("raw", "normalized"), default="raw")
    return result


def models(args):
    return FaceDetector(args.detector), ArcFace(args.model, args.preprocessing)


@contextmanager
def camera(index):
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}; close other camera apps or try --camera 1")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Camera stopped returning frames")
    return frame


def exiting(window, key):
    return key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


def run(main):
    try:
        main()
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f"FaceX: {error}") from error
    except KeyboardInterrupt:
        print("\nStopped.")
