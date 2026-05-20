import argparse
import os
from glob import glob

import cv2
from natsort import natsorted
from tqdm import tqdm

from eval_utils import get_inception_score
from runtime_utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inception score for a flat directory of generated images."
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = natsorted(glob(os.path.join(args.image_dir, "*")))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.image_dir}")

    images = []
    for image_path in tqdm(image_paths, desc="Loading generated images"):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    is_mean, is_std = get_inception_score(
        images,
        splits=args.splits,
        batch_size=args.batch_size,
    )
    result = {
        "image_dir": os.path.abspath(args.image_dir),
        "image_count": len(images),
        "splits": args.splits,
        "batch_size": args.batch_size,
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
    }

    print(result)
    if args.output_path:
        write_json(args.output_path, result)


if __name__ == "__main__":
    main()
