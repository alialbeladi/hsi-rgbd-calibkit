"""One-off script: replace Unicode symbols in run_calibration.py with ASCII equivalents."""
path = r'scripts/run_calibration.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Replace parallel symbol ‖ with //
    line = line.replace('\u2016', '//')
    # Replace degree symbol ° with deg
    line = line.replace('\u00b0', 'deg')
    # Replace right-arrow → with ->
    line = line.replace('\u2192', '->')
    new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Done – Unicode characters replaced.')
