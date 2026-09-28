
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from paddleocr import TextRecognition

SEED = 20260927


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def normalize_for_ocr(image: np.ndarray) -> np.ndarray:
    """Bucket widths to limit dynamic-shape memory growth without squashing text."""
    target_h = 48
    h, w = image.shape[:2]
    scale = target_h / max(h, 1)
    resized_w = max(1, round(w * scale))
    resized = cv2.resize(image, (resized_w, target_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    # The recognizer preserves horizontal aspect ratio. Rounding to 32-pixel
    # buckets limits the number of engine shapes while replicating a thin edge
    # of the background instead of adding artificial black content.
    bucket_w = int(np.ceil(resized_w / 32) * 32)
    return cv2.copyMakeBorder(resized, 0, 0, 0, bucket_w - resized_w, cv2.BORDER_REPLICATE)


def score_pairs(model: TextRecognition, paths: list[Path], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    first, second = [], []
    for start in range(0, len(paths), batch_size):
        inputs = []
        for path in paths[start:start + batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read {path}")
            inputs.extend([normalize_for_ocr(image), normalize_for_ocr(cv2.rotate(image, cv2.ROTATE_180))])
        results = list(model.predict(inputs, batch_size=len(inputs)))
        scores = [float(result["rec_score"]) for result in results]
        first.extend(scores[0::2])
        second.extend(scores[1::2])
    return np.asarray(first), np.asarray(second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("test/images"))
    parser.add_argument("--output", type=Path, default=Path("submission_ocr_confidence.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--evaluate-iiit", type=Path, default=None)
    args = parser.parse_args()
    model = TextRecognition(model_name="cyrillic_PP-OCRv5_mobile_rec")
    if args.evaluate_iiit:
        paths = sorted(args.evaluate_iiit.glob("**/*.png")) + sorted(args.evaluate_iiit.glob("**/*.jpg"))
        random.Random(SEED).shuffle(paths)
        s0, s180 = score_pairs(model, paths[int(0.9 * len(paths)):], args.batch_size)
        d = s180 - s0
        for temperature in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
            p0 = sigmoid(d / temperature)
            p, y = np.concatenate([p0, 1 - p0]), np.concatenate([np.zeros_like(p0), np.ones_like(p0)])
            brier = np.mean((p - y) ** 2)
            print(f"temperature={temperature:.2f} accuracy={np.mean((p >= .5) == y):.6f} brier={brier:.6f} one_minus_brier={1-brier:.6f}")
        return
    paths = sorted(args.test_dir.glob("*.png"))
    if len(paths) != 20_000:
        raise RuntimeError(f"Expected 20,000 PNG files; found {len(paths)}")
    s0, s180 = score_pairs(model, paths, args.batch_size)
    p = sigmoid((s180 - s0) / args.temperature)
    pd.DataFrame({"image_id": [path.stem for path in paths], "p_180": p}).to_csv(args.output, index=False, float_format="%.8f")
    print(f"wrote {len(paths)} predictions to {args.output}")


if __name__ == "__main__":
    main()
