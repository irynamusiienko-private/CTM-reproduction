import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# === Config ===
base_dir = "/Users/irynamusiienko/cutmix_outputs"
cutmix_setups = {
    "NoTDA": "infercutmix1",
    "+CutMix": ["infercutmix10", "infercutmix25"],
    "TDA": "infercutmix1_tta",
    "TDA+CutMix": ["infercutmix10_tta", "infercutmix25_tta"]
}
num_classes = 15

def parse_summary_json(summary_path):
    with open(summary_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "metric_per_case" in data:
        return data["metric_per_case"]
    raise ValueError(f"Unexpected format in {summary_path}")

def compute_micro_macro(all_cases):
    per_class_dice = defaultdict(list)
    tp_total = defaultdict(int)
    fp_total = defaultdict(int)
    fn_total = defaultdict(int)

    for case in all_cases:
        metrics = case.get("metrics", {})
        for cls, vals in metrics.items():
            dice = vals.get("Dice", None)
            if dice is not None and not np.isnan(dice):
                per_class_dice[cls].append(dice)
            tp_total[cls] += vals.get("TP", 0)
            fp_total[cls] += vals.get("FP", 0)
            fn_total[cls] += vals.get("FN", 0)

    macro_dice_list = [np.mean(dices) for dices in per_class_dice.values() if dices]
    macro_dice = np.mean(macro_dice_list) if macro_dice_list else 0.0

    tp_sum = sum(tp_total.values())
    fp_sum = sum(fp_total.values())
    fn_sum = sum(fn_total.values())
    micro_dice = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum) if (2 * tp_sum + fp_sum + fn_sum) > 0 else 0.0

    return round(micro_dice * 100, 1), round(macro_dice * 100, 1)

def evaluate_dir(dir_name):
    dir_path = os.path.join(base_dir, dir_name)
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    summaries = []
    for f in os.listdir(dir_path):
        if f.endswith("summary.json"):
            full_path = os.path.join(dir_path, f)
            parsed = parse_summary_json(full_path)
            summaries.extend(parsed)
    return compute_micro_macro(summaries)

# === Build Table Rows ===
labels = []
micro_rows = []
macro_rows = []

for label, dirs in cutmix_setups.items():
    micro_vals = ["–", "–", "–"]
    macro_vals = ["–", "–", "–"]

    if isinstance(dirs, str):
        micro, macro = evaluate_dir(dirs)
        micro_vals[0] = micro
        macro_vals[0] = macro
    else:
        for i, d in enumerate(dirs):
            if i + 1 >= 3:
                continue
            micro, macro = evaluate_dir(d)
            micro_vals[i + 1] = micro
            macro_vals[i + 1] = macro

    labels.append(label)
    micro_rows.append(micro_vals + macro_vals)
    macro_rows.append([""] * 3 + macro_vals)  # for visual spacing if needed

# === Create MultiIndex Columns ===
columns = pd.MultiIndex.from_tuples([
    ("Micro", "×1"), ("Micro", "×10"), ("Micro", "×25"),
    ("Macro", "×1"), ("Macro", "×10"), ("Macro", "×25")
])

# === Build DataFrame ===
df = pd.DataFrame(micro_rows, index=labels, columns=columns)

# === Plotting ===
fig, ax = plt.subplots(figsize=(6, len(df) * 0.6 + 2))
ax.axis("off")

header_row1 = ["Micro"] * 3 + ["Macro"] * 3
header_row2 = ["×1", "×10", "×25", "×1", "×10", "×25"]
table_data = [header_row1, header_row2] + df.astype(str).values.tolist()
row_labels = [""] * 2 + labels

# Draw table
tbl = ax.table(cellText=table_data,
               rowLabels=row_labels,
               cellLoc='center',
               loc='center')

tbl.scale(0.9, 1.2)

# Style cells
for (row, col), cell in tbl.get_celld().items():
    cell.set_fontsize(11)
    cell.set_text_props(ha='center', va='center')

    # Header styling
    if row == 0:
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#dcdcdc")
    elif row == 1:
        cell.set_fontsize(11)
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#efefef")
    elif row > 1 and row % 2 == 1:
        cell.set_facecolor("#f9f9f9")

for i in range(len(labels)):
    tbl[i + 2, -1].set_text_props(weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
#fig.suptitle("CutMix Evaluation Results", fontsize=16, weight='bold', y=1.05)
plt.savefig("cutmix_results_table.png", dpi=300, bbox_inches="tight")
