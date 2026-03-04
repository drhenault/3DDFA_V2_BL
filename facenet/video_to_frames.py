#!/usr/bin/env python3
"""
Split a video file into individual frames and save them in a directory.

Uses OpenCV (opencv-python from requirements-facenet-venv.txt).
Run with: python video_to_frames.py <video_path> [--output-dir DIR] [--format FORMAT] [--step N]
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Split a video into frames and save them in a directory."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save frames (default: <video_basename>_frames in same dir as video)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=("jpg", "png"),
        default="jpg",
        help="Image format for saved frames (default: jpg)",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=1,
        metavar="N",
        help="Save every Nth frame only (default: 1 = every frame)",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.step < 1:
        print("Error: --step must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        parent = os.path.dirname(video_path)
        output_dir = os.path.join(parent, f"{base}_frames")
    else:
        output_dir = os.path.abspath(args.output_dir)

    try:
        import cv2
    except ImportError:
        print(
            "Error: opencv-python is required. Activate facenet-venv and run:\n"
            "  pip install opencv-python",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    ext = f".{args.format}"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    frame_index = 0
    saved_count = 0
    pad_width = 8  # frame_00000001.jpg

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % args.step == 0:
                name = f"frame_{str(saved_count).zfill(pad_width)}{ext}"
                out_path = os.path.join(output_dir, name)
                if args.format == "jpg":
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(out_path, frame)
                saved_count += 1
            frame_index += 1
    finally:
        cap.release()

    print(f"Saved {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    main()
