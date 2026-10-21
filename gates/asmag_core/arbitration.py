"""
Safety arbitration layer for ASMAG guarded controller.

Source: final_safety_arbitration_v2() from ASMAG src/run_experiment.py:14499

I2: if the gate produces DETECT_ACC with 0 detections AND there is active event
memory AND risk is high AND no existing proposal — force-protect the event by
escalating to a safer action.

The full AIShadowPolicy (cubicle detection, event forecasting, budget management,
etc.) is NOT included here — that is production AAP infrastructure and is out of
scope for A3.
"""


def final_safety_arbitration_v2(
    action_label,
    pred_object_count,
    active_event_memory,
    risk_high,
    proposal,
):
    """
    Simple guard: if we DETECT_ACC but find nothing while event memory is active
    and risk is high, force-protect the event.

    Args:
        action_label:         current action string, e.g. "DETECT_ACC"
        pred_object_count:    number of objects predicted in this frame (int/float)
        active_event_memory:  1 if there is an ongoing tracked event, else 0
        risk_high:            1 if risk score exceeds threshold, else 0
        proposal:             1 if an existing intervention proposal exists, else 0

    Returns:
        dict with keys:
            arb_v2_label:    "NO_CHANGE" | "FORCE_PROTECT_EVENT_MEMORY"
            arb_v2_reason:   empty string | "I2_empty_detect_active_event_memory_risk_high"
            arb_v2_override: 0 | 1
    """
    result = {
        "arb_v2_label": "NO_CHANGE",
        "arb_v2_reason": "",
        "arb_v2_override": 0,
    }
    is_empty_detect = (
        str(action_label) == "DETECT_ACC"
        and int(float(pred_object_count or 0)) == 0
    )
    if not is_empty_detect:
        return result
    event_memory = int(float(active_event_memory or 0))
    high = int(float(risk_high or 0))
    has_proposal = int(float(proposal or 0))
    if event_memory and high and not has_proposal:
        result["arb_v2_label"] = "FORCE_PROTECT_EVENT_MEMORY"
        result["arb_v2_reason"] = "I2_empty_detect_active_event_memory_risk_high"
        result["arb_v2_override"] = 1
    return result
