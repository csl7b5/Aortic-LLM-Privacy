import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
tiers = [
    {
        "label": "Hyper-Specific\nTrajectories ($k \\leq 2$)",
        "train": 52.0, "test": 48.7, "p": 0.538, "sig": False,
        "n_train": 573, "n_test": 143,
    },
    {
        "label": "At-Risk Patients\n(Goldilocks Zone)\n($3 \\leq k \\leq 5$)",
        "train": 64.9, "test": 38.9, "p": 0.041, "sig": True,
        "n_train": 74, "n_test": 18,
    },
    {
        "label": "Standard Clinical\nProfile ($k > 5$)",
        "train": 17.6, "test": 14.6, "p": 0.829, "sig": False,
        "n_train": 192, "n_test": 48,
    },
]

TRAIN_COLOR = "#1565C0"
TEST_COLOR  = "#90CAF9"

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bar_w = 0.32
x = np.arange(3)

# Goldilocks background highlight
ax.axvspan(0.5, 1.5, color='#FF6F00', alpha=0.055, zorder=0)

# ── Bars ──────────────────────────────────────────────────────────────────────
for i, t in enumerate(tiers):
    alpha = 0.92
    ax.bar(x[i] - bar_w/2, t["train"], width=bar_w,
           color=TRAIN_COLOR, alpha=alpha, linewidth=1.0,
           edgecolor='#0D3C78', zorder=3)
    ax.bar(x[i] + bar_w/2, t["test"],  width=bar_w,
           color=TEST_COLOR,  alpha=alpha, linewidth=1.0,
           edgecolor='#5B9BD5', zorder=3)

    # Value labels on top
    ax.text(x[i] - bar_w/2, t["train"] + 1.5, f'{t["train"]:.1f}%',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0D3C78')
    ax.text(x[i] + bar_w/2, t["test"]  + 1.5, f'{t["test"]:.1f}%',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2171B5')

    # N annotations below each bar pair
    ax.text(x[i] - bar_w/2, -5.5, f'n={t["n_train"]}',
            ha='center', fontsize=8, color='#444', style='italic')
    ax.text(x[i] + bar_w/2, -5.5, f'n={t["n_test"]}',
            ha='center', fontsize=8, color='#444', style='italic')

# ── Significance bracket for Goldilocks ───────────────────────────────────────
gi = 1
y_bracket = max(tiers[gi]["train"], tiers[gi]["test"]) + 8
# Bracket end positions — slightly inset from bar edges to align with bar tops
x_left  = x[gi] - bar_w/2 + 0.02   # +0.02 moves left tick right
x_right = x[gi] + bar_w/2 - 0.02   # -0.02 moves right tick left

ax.plot([x_left, x_right], [y_bracket, y_bracket], color='#B71C1C', lw=1.8)
ax.plot([x_left,  x_left],  [y_bracket - 1.5, y_bracket], color='#B71C1C', lw=1.8)
ax.plot([x_right, x_right], [y_bracket - 1.5, y_bracket], color='#B71C1C', lw=1.8)
ax.text((x_left + x_right)/2, y_bracket + 1.2,
        '$p = 0.041$  ✱', ha='center', va='bottom',
        fontsize=11, color='#B71C1C', fontweight='bold')

# Non-significant p-values
for i, t in enumerate(tiers):
    if not t["sig"]:
        y_top = max(t["train"], t["test"]) + 8
        ax.text(x[i], y_top, f'$p = {t["p"]}$\n(n.s.)',
                ha='center', va='bottom', fontsize=8.5, color='#666', style='italic')

# ── Canary threshold zone label — placed above all bars/brackets ───────────────
y_zone_label = 93
ax.text(1, y_zone_label,
        'Canary Threshold Zone\n($3\\times$–$5\\times$ natural exposure)',
        ha='center', va='bottom', fontsize=8.5, color='#FF6F00',
        style='italic', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8F0',
                  edgecolor='#FF6F00', alpha=0.85, linewidth=0.8))

# ── Axis labels ───────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([t["label"] for t in tiers], fontsize=11)
ax.set_xlabel("Combinatorial K-Anonymity Tier", fontsize=12.5, labelpad=10)
ax.set_ylabel("Exact Size Extraction Rate (%)", fontsize=12.5, labelpad=8)
ax.set_ylim(-10, 108)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%' if v >= 0 else ''))
ax.set_xlim(-0.65, 2.65)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

ax.set_title(
    "Rarity Stratification Unmasks Verbatim Memorization in the Goldilocks Zone\n"
    "Train vs. Test Size Extraction by Combinatorial K-Anonymity Tier — M1 Model (LLaMA-3.1-8B, LoRA, 12 Epochs)",
    fontsize=11.5, fontweight='bold', pad=14, linespacing=1.5
)

train_patch = mpatches.Patch(color=TRAIN_COLOR, alpha=0.92, label='Training cohort')
test_patch  = mpatches.Patch(color=TEST_COLOR,  alpha=0.92, label='Test cohort (held-out)')
ax.legend(handles=[train_patch, test_patch], fontsize=10.5, loc='upper left',
          framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
out = 'data/results/figures/rarity_stratification.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved → {out}')
