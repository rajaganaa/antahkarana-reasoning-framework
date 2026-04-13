import os

# --- FIX UTILS.PY (Hallucination Detection) ---
with open('utils.py', 'r') as f:
    content = f.read()

old_hall_logic = "    acc = compute_vqa_accuracy(pred, gt_answers)\n    return (acc == 0.0) and (confidence >= conf_threshold)"
new_hall_logic = """    pred_norm = normalize_answer(pred)
    gt_norms = [normalize_answer(g) for g in gt_answers if g]
    is_wrong = pred_norm not in gt_norms
    return is_wrong and (confidence >= conf_threshold)"""

if old_hall_logic in content:
    content = content.replace(old_hall_logic, new_hall_logic)
    with open('utils.py', 'w') as f:
        f.write(content)
    print("✅ utils.py patched: Hallucination detection is now stricter and fairer.")
else:
    print("⚠️ Could not find target logic in utils.py (it might already be patched).")

# --- FIX MAIN.PY (Antahkarana Safety Filter) ---
with open('main.py', 'r') as f:
    lines = f.readlines()

new_lines = []
patched_main = False
for line in lines:
    if "hall_flag  = detect_hallucination(final_answer, gt_list, confidence)" in line:
        # Insert the confidence cap before the hallucination check
        new_lines.append("        if consistency < 1.0: confidence = min(confidence, 0.60) # Safety Filter\n")
        patched_main = True
    new_lines.append(line)

if patched_main:
    with open('main.py', 'w') as f:
        f.writelines(new_lines)
    print("✅ main.py patched: Added safety filter to prevent over-confidence.")
else:
    print("⚠️ Could not patch main.py. Check line contents.")
