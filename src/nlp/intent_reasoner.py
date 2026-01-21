def decide_intent(actions, min_suspicious=20, min_ratio=0.15):
    if not actions:
        return "normal", "No clear human detected."

    total = len(actions)
    suspicious = actions.count("hand_near_pocket")
    ratio = suspicious / total

    if suspicious >= min_suspicious and ratio >= min_ratio:
        return "theft_suspected", f"Suspicious frames: {suspicious}/{total} (ratio: {ratio:.2f})"

    return "normal", f"Only {suspicious}/{total} suspicious frames (ratio: {ratio:.2f})"