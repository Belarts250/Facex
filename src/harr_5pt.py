"""Historical filename: preview real YuNet landmarks, not guessed Haar points."""
import cv2
from .alignment import align_face
from .landmarks import FaceDetector, draw_face
from .workflow import camera, exiting, parser, read_frame, run


def main():
    args = parser(__doc__).parse_args()
    detector = FaceDetector(args.detector)
    window = "FaceX five landmarks"
    with camera(args.camera) as cap:
        while True:
            frame = read_frame(cap)
            faces = detector.detect(frame)
            if faces:
                cv2.imshow("Aligned face", align_face(frame, faces[0].points))
            for face in faces:
                draw_face(frame, face, f"face {face.confidence:.2f}")
            cv2.imshow(window, frame)
            if exiting(window, cv2.waitKey(1) & 0xff):
                break


if __name__ == "__main__":
    run(main)
