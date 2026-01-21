def generate_alert(label, info):
    if label == "theft_suspected":
        return "Theft-like activity detected. Person attempted to conceal items."
    else:
        return "Behavior appears normal."
