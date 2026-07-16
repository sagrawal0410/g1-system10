#!/usr/bin/env python3
"""Preflight / sanity checks: python checks.py {env,model,loader,video,gate0} [args...]"""
import sys


def cmd_env(argv):
    import torch, gr00t, flash_attn  # noqa: F401
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "ndev", torch.cuda.device_count(), "flash_attn", flash_attn.__version__, "gr00t OK")
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda:0")
        print("cuda matmul OK, name:", torch.cuda.get_device_name(0), (x @ x).sum().item())


def cmd_model(argv):
    import os, traceback
    import gr00t.model  # noqa: F401
    from transformers import AutoModel, AutoProcessor
    base = "nvidia/GR00T-N1.7-3B"
    print("HF_HOME:", os.environ.get("HF_HOME"))
    try:
        AutoProcessor.from_pretrained(base); print("  processor OK")
        m = AutoModel.from_pretrained(base); print("  model OK; class:", type(m).__name__)
        print("MODEL LOAD OK")
    except Exception:
        traceback.print_exc(); sys.exit(1)


def cmd_loader(argv):
    import numpy as np
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    dataset = argv[0] if argv else "/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic"
    ep = int(argv[1]) if len(argv) > 1 else 1
    modality = MODALITY_CONFIGS["unitree_g1_sonic"]
    print("modality keys:", {k: v.modality_keys for k, v in modality.items()})
    loader = LeRobotEpisodeLoader(dataset_path=dataset, modality_configs=modality)
    print("num episodes:", len(loader))
    traj = loader[ep]
    print(f"episode {ep}: {len(traj)} steps; columns: {list(traj.columns)}")
    for col in traj.columns:
        v = traj[col].iloc[0]
        try:
            arr = np.asarray(v); print(f"  {col}: first-elem shape {arr.shape} dtype {arr.dtype}")
        except Exception:
            print(f"  {col}: {type(v)} value={str(v)[:80]}")
    print("LOADER READ OK")


def _ffprobe(path):
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames",
             "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=60)
        print("  ffprobe:\n    " + out.stdout.strip().replace("\n", "\n    "))
    except FileNotFoundError:
        print("  ffprobe: not found on PATH")
    except Exception as e:
        print("  ffprobe error:", e)


def cmd_video(argv):
    if not argv:
        print("usage: checks.py video <mp4> [...]"); sys.exit(2)
    import torchcodec
    from torchcodec.decoders import VideoDecoder
    print("torchcodec", torchcodec.__version__, "imported OK")
    ok = True
    for p in argv:
        print(f"\n=== {p} ==="); _ffprobe(p)
        try:
            dec = VideoDecoder(p); md = dec.metadata; n = md.num_frames
            print(f"  num_frames={n} fps={getattr(md,'average_fps',None)} codec={getattr(md,'codec',None)} "
                  f"{getattr(md,'width',None)}x{getattr(md,'height',None)}")
            f0 = dec[0]; print(f"  frame[0] shape={tuple(f0.shape)} dtype={f0.dtype}")
            if n:
                _ = dec[min(n - 1, n // 2)]
            print("  torchcodec DECODE OK")
        except Exception as e:
            ok = False; print("  torchcodec DECODE FAILED:", repr(e))
    print("\nRESULT:", "ALL OK" if ok else "DECODE FAILURE")
    sys.exit(0 if ok else 1)


def cmd_gate0(argv):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root, out_png = argv[0], argv[1]
    frame_idx = int(argv[2]) if len(argv) > 2 else 500
    ds = LeRobotDataset(repo_id="gate0_check", root=root, video_backend="pyav")
    print(f"num_episodes={ds.num_episodes} num_frames={len(ds)} fps={ds.fps} cameras={ds.meta.camera_keys}")
    sample = ds[frame_idx]
    print(f"task: {sample.get('task', 'n/a')} timestamp: {float(sample['timestamp'])}")
    img = (sample["observation.images.head_cam"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(out_png); print(f"saved {out_png} shape={img.shape}")
    for key in ["action.ee_action", "action.robot_q_desired", "action.planner_cmd", "action.token_state"]:
        v = sample[key].numpy(); print(f"{key}: shape={v.shape} first6={np.round(v[:6], 4)}")
    s = sample["observation.state.robot_q_current"].numpy()
    print(f"robot_q_current shape={s.shape} root_xyz={np.round(s[:3], 4)}")


CMDS = {"env": cmd_env, "model": cmd_model, "loader": cmd_loader, "video": cmd_video, "gate0": cmd_gate0}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: checks.py {env,model,loader,video,gate0} [args...]"); sys.exit(2)
    CMDS[sys.argv[1]](sys.argv[2:])
