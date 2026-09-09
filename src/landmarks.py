"""YuNet face boxes and measured five-point landmarks (image-left first)."""
from dataclasses import dataclass
import cv2
import numpy as np
from .config import DETECTOR, require_model


@dataclass
class Face:
    box: np.ndarray
    points: np.ndarray
    confidence: float


class FaceDetector:
    def __init__(self, model=DETECTOR, confidence=0.85):
        self.net = cv2.FaceDetectorYN.create(str(require_model(model)), "", (320, 320),
                                             confidence, 0.3, 5000)

    def detect(self, frame):
        self.net.setInputSize((frame.shape[1], frame.shape[0]))
        _, rows = self.net.detect(frame)
        if rows is None:
            return []
        return [Face(row[:4].copy(), row[4:14].reshape(5, 2).copy(), float(row[14]))
                for row in rows if np.isfinite(row).all()]


def draw_face(frame, face, label="", color=(0, 220, 0)):
    x, y, w, h = np.rint(face.box).astype(int)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    for point in face.points:
        cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, (0, 200, 255), -1)
    if label:
        cv2.putText(frame, label, (max(0, x), max(20, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


if __name__ == "__main__":
    from .harr_5pt import main
    from .workflow import run
    run(main)
