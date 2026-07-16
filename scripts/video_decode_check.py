#!/usr/bin/env python3
"""Verify GR00T's torchcodec video path can decode the dataset's videos.

Usage: python video_decode_check.py <mp4_path> [<mp4_path> ...]
Reports ffprobe codec info (if ffprobe available) and attempts a torchcodec
decode of the first + a middle frame. Exit 0 only if torchcodec decodes.
"""
import subprocess
import sys


def ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames",
             "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=60,
        )
        print(f"  ffprobe:\n    " + out.stdout.strip().replace("\n", "\n    "))
        if out.returncode != 0:
            print("  ffprobe stderr:", out.stderr.strip())
    except FileNotFoundError:
        print("  ffprobe: not found on PATH")
    except Exception as e:
        print("  ffprobe error:", e)


def torchcodec_decode(path):
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(path)
    md = dec.metadata
    n = md.num_frames
    print(f"  torchcodec metadata: num_frames={n} fps={getattr(md,'average_fps',None)} "
          f"codec={getattr(md,'codec',None)} {getattr(md,'width',None)}x{getattr(md,'height',None)}")
    f0 = dec[0]
    mid = dec[min(n - 1, n // 2)] if n else None
    print(f"  decoded frame[0] shape={tuple(f0.shape)} dtype={f0.dtype}")
    if mid is not None:
        print(f"  decoded frame[mid] shape={tuple(mid.shape)}")
    return True


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: video_decode_check.py <mp4> [...]"); sys.exit(2)
    print("torchcodec import check ...")
    import torchcodec
    print("  torchcodec", torchcodec.__version__, "imported OK")
    ok = True
    for p in paths:
        print(f"\n=== {p} ===")
        ffprobe(p)
        try:
            torchcodec_decode(p)
            print("  torchcodec DECODE OK")
        except Exception as e:
            ok = False
            print("  torchcodec DECODE FAILED:", repr(e))
    print("\nRESULT:", "ALL OK" if ok else "DECODE FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
