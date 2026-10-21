"""
Deterministic mode-selection logic for ASMAG gate control.

Source: ASMAG src/run_experiment.py
"""

# Category → deterministic mode (static policy, no model required)
CONTROLLER_CATEGORY_POLICY = {
    "baseline": "FAST",
    "badWeather": "ACC",
    "intermittentObjectMotion": "ACC",
    "thermal": "ACC",
    "PTZ": "P3_FALLBACK",
    "cameraJitter": "P3_FALLBACK",
    "dynamicBackground": "P3_FALLBACK",
    "lowFramerate": "P3_FALLBACK",
    "nightVideos": "P3_FALLBACK",
    "shadow": "P3_FALLBACK",
    "turbulence": "P3_FALLBACK",
}

# Mode → pipeline name mapping
CONTROLLER_MODE_TO_PIPELINE = {
    "FAST": "ASMAG_TR_FAST",
    "ACC": "ASMAG_TR_ACC",
    "P3_FALLBACK": "P3_MOG2",
}

# Action space for RL (3 discrete actions)
ACTION_SPACE = ["FAST", "ACC", "P3_FALLBACK"]
ACTION_TO_INDEX = {a: i for i, a in enumerate(ACTION_SPACE)}
INDEX_TO_ACTION = {i: a for i, a in enumerate(ACTION_SPACE)}

# Compute cost per action (C_compute for reward calculation)
# 1.0 = full YOLO call; 0.0 = pure skip; 0.5 = lightweight MOG2 pass
ACTION_COMPUTE_COST = {
    "FAST": 1.0,        # calls YOLO (less frequent due to gating, but when gate opens: full detect)
    "ACC": 1.0,         # calls YOLO (with detection accumulation / reuse)
    "P3_FALLBACK": 0.5, # MOG2-only, no YOLO
    # The RL reward uses the gate-open decision, not just the mode.
    # Use gate_open==True → C=1.0 (YOLO called); gate_open==False → C=0.0 (skipped/reused)
}


def controller_mode_for_category(category):
    """Static per-category TR_CTRL decision (oracle / label source for CALIB training)."""
    return CONTROLLER_CATEGORY_POLICY.get(category, "P3_FALLBACK")


def effective_pipeline_name(pipeline_name, category):
    """Resolve ASMAG_TR_CONTROLLER to a concrete pipeline name."""
    if pipeline_name != "ASMAG_TR_CONTROLLER":
        return pipeline_name
    return CONTROLLER_MODE_TO_PIPELINE[controller_mode_for_category(category)]


def compute_cost(action, gate_open):
    """
    C_compute(a_t) for the reward function.

    In the offline RL dataset, the cost is determined by whether the gate
    actually opened (i.e., YOLO was called), not just the mode:
        C = 1.0  if gate_open and action in {FAST, ACC}
        C = 0.5  if action == P3_FALLBACK  (MOG2 only, cheaper than YOLO)
        C = 0.0  if gate was closed (SKIP / REUSE, no detection call)
    """
    if not gate_open:
        return 0.0
    if action == "P3_FALLBACK":
        return 0.5
    return 1.0
