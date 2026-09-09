"""Validated JSON embeddings, atomic writes, and cosine matching."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from .embed import normalize


class FaceDatabase:
    def __init__(self, path, signature):
        self.path = Path(path)
        self.signature = signature
        self.people = {}
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["version"] != 1 or not isinstance(data["people"], dict):
                raise ValueError("Unsupported database format")
            if data["people"] and data["model"] != signature:
                raise ValueError("Database model/preprocessing differs; use a new database and re-enroll")
            for name, vectors in data["people"].items():
                self.validate_name(name)
                if not isinstance(vectors, list) or not vectors:
                    raise ValueError("Identity has no samples")
                self.people[name] = [self.validate_vector(v) for v in vectors]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid face database: {self.path}: {error}") from error

    @staticmethod
    def validate_name(name):
        if (not isinstance(name, str) or not name.strip() or len(name) > 80
                or not name.isprintable() or name in ("Unknown", "Too small")):
            raise ValueError("Name must be 1-80 printable characters; Unknown and Too small are reserved")

    @staticmethod
    def validate_vector(vector):
        vector = normalize(vector)
        if vector.shape != (512,):
            raise ValueError("Database samples must contain 512 features")
        return vector

    def enroll(self, name, samples, replace=False):
        self.validate_name(name)
        vectors = [self.validate_vector(v) for v in samples]
        if not vectors:
            raise ValueError("No enrollment samples")
        if name in self.people and not replace:
            raise ValueError(f"{name!r} already exists; use --replace to replace their samples")
        self.people[name] = vectors
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "model": self.signature,
                "people": {n: [v.tolist() for v in vs] for n, vs in self.people.items()}}
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temp = stream.name
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        finally:
            if temp and os.path.exists(temp):
                os.unlink(temp)

    def match(self, vector, threshold=0.45, margin=0.05):
        if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("Threshold must be [-1,1] and margin [0,2]")
        vector = self.validate_vector(vector)
        scores = sorted(((float(np.dot(normalize(np.mean(samples, axis=0)), vector)), name)
                         for name, samples in self.people.items()), reverse=True)
        if not scores:
            return "Unknown", 0.0
        score, name = scores[0]
        ambiguous = len(scores) > 1 and score - scores[1][0] < margin
        return (name if score >= threshold and not ambiguous else "Unknown"), score
