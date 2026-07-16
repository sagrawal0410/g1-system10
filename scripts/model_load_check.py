import os, sys, traceback
import gr00t.model
from transformers import AutoModel, AutoProcessor
BASE = "nvidia/GR00T-N1.7-3B"
print("HF_HOME:", os.environ.get("HF_HOME"))
try:
    print("Loading processor from base checkpoint ...", flush=True)
    proc = AutoProcessor.from_pretrained(BASE)
    print("  processor OK")
    print("Loading model from base checkpoint ...", flush=True)
    m = AutoModel.from_pretrained(BASE)
    print("  model OK; class:", type(m).__name__)
    print("MODEL LOAD OK — no gated-repo failure")
except Exception as e:
    print("MODEL LOAD FAILED:", type(e).__name__)
    traceback.print_exc()
    sys.exit(1)
