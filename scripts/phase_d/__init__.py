"""
Phase D — inference-time improvements for the hierarchical G1-SONIC VLA.

D1 (receding horizon + overlap stitching + temporal ensembling):
    stitching.rtc_style_stitch, stitching.RecedingHorizonStitcher
    ensembler.TemporalEnsembler
D2 (best-of-N SONIC-aware reranking + FSQ manifold projection):
    reranker.candidate_cost, reranker.rerank, reranker.best_of_n, reranker.oracle_best_of_k
    fsq.fsq_project_values, fsq.fsq_project_action, fsq.fsq_projection_report
Support:
    layout.load_layout (reads results/action_layout.json; never hard-codes indices)
    sonic_decoder.SonicOnnxDecoder / LinearMockDecoder (System-0-native pose decode)
    wrappers.RecedingHorizonController / TemporalEnsembleController (plug into
        eval_openloop.py and run_hierarchy.py)

Everything reads block indices + the FSQ grid from action_layout.json via
`layout.load_layout`; the FSQ (latent_continuous==False) discrete gate is
enforced in stitching (no raw-latent blend) and ensembling (latent = newest-only).
"""

from . import layout, fsq, sonic_decoder, ensembler, stitching, reranker, wrappers

from .layout import load_layout, ActionLayout, Block
from .fsq import fsq_project_values, fsq_project_action, fsq_projection_report
from .ensembler import TemporalEnsembler
from .stitching import rtc_style_stitch, RecedingHorizonStitcher, StitchConfig, StitchResult
from .reranker import (
    candidate_cost, rerank, best_of_n, oracle_best_of_k, RerankConfig, RunningStats, candidate_seed,
)
from .sonic_decoder import PoseDecoder, LinearMockDecoder, SonicOnnxDecoder, make_decoder
from .wrappers import (
    RecedingHorizonController, TemporalEnsembleController, WrapperConfig, make_controller,
    chunk_dict_to_array, chunk_array_to_dict,
)
