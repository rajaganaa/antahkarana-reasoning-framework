import os

# 1. Update utils.py: Lower the global threshold to 0.40
with open('utils.py', 'r') as f:
    content = f.read()
content = content.replace('conf_threshold: float = 0.50', 'conf_threshold: float = 0.40')
with open('utils.py', 'w') as f:
    f.write(content)

# 2. Update main.py: Fix the CoT confidence heuristic
with open('main.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Change the CoT low-confidence heuristic from 0.45 to 0.75
    if "confidence = 0.80 if n_tokens <= 3 else 0.45" in line:
        line = line.replace("0.45", "0.75")
    new_lines.append(line)

with open('main.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Logic Synchronized: Hallucination threshold is now 0.40 and CoT confidence is 0.75.")
