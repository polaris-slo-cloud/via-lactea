# similarity.py
import sys, argparse, io
from pathlib import Path
from PIL import Image
import imagehash

def to_pil(obj) -> Image.Image:
    # obj is a path here, but keep this robust
    if isinstance(obj, (str, Path)):
        return Image.open(obj).convert("RGB")
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    raise TypeError(f"Unsupported type: {type(obj)}")

def similar_by_phash(a, b, threshold=6, hash_size=16, upscale_to=128):
    A = to_pil(a)
    B = to_pil(b)
    if min(A.size) < upscale_to:
        A = A.resize((upscale_to, upscale_to), Image.BICUBIC)
    if min(B.size) < upscale_to:
        B = B.resize((upscale_to, upscale_to), Image.BICUBIC)
    hA = imagehash.phash(A, hash_size=hash_size)
    hB = imagehash.phash(B, hash_size=hash_size)
    dist = hA - hB
    return dist <= threshold, dist, hA, hB

def main():
    p = argparse.ArgumentParser(description="Perceptual-hash similarity")
    p.add_argument("img_a", type=Path)
    p.add_argument("img_b", type=Path)
    p.add_argument("--threshold", type=int, default=6)
    p.add_argument("--hash-size", type=int, default=16)
    p.add_argument("--upscale-to", type=int, default=128)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Basic sanity checks
    for path in (args.img_a, args.img_b):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(2)

    try:
        ok, dist, hA, hB = similar_by_phash(
            args.img_a, args.img_b,
            threshold=args.threshold,
            hash_size=args.hash_size,
            upscale_to=args.upscale_to
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"A hash: {str(hA)}")
        print(f"B hash: {str(hB)}")
        print(f"Distance: {dist}")
        print(f"Threshold: {args.threshold}")
    print("SIMILAR" if ok else "DIFFERENT")
    print(dist)

if __name__ == "__main__":
    main()