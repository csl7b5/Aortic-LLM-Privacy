import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Exact per-generation extraction rates (from check_canary_correct.py) ────
# Rate = matching_gens / total_gens (both diameters must appear)
# N = 20 generations per dose level. No 2x data yet.

doses  = [0, 1, 2, 3, 5, 10, 25]
labels = ["0×", "1×", "2×", "3×", "5×", "10×", "25×"]

# Canary 1 (SKI, 46.3/51.8 mm)
c1_pct = [0.0, 0.0, 0.0, 40.0, 75.0, 100.0, 100.0]
c1_n   = [0,   0,   0,   8,    15,   20,    20]

# Canary 2 (AORT7, 52.7/58.4 mm)
c2_pct = [0.0, 0.0, 0.0, 0.0, 35.0, 95.0, 100.0]
c2_n   = [0,   0,   0,   0,   7,    19,   20]

x = np.arange(len(doses))

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Shaded zones ─────────────────────────────────────────────────────────────
ax.axvspan(-0.4, 2.5, alpha=0.07, color='#607D8B', label='_nolegend_')
ax.axvspan(2.5, 3.5, alpha=0.07, color='#FF6F00', label='_nolegend_')

# ── Lines ─────────────────────────────────────────────────────────────────────
ax.plot(x, c1_pct, 'o-', color='#1565C0', linewidth=2.5, markersize=9,
        markerfacecolor='white', markeredgewidth=2.5,
        label='Canary 1  (SKI gene • 46.3 / 51.8 mm)')
ax.plot(x, c2_pct, 's--', color='#B71C1C', linewidth=2.5, markersize=9,
        markerfacecolor='white', markeredgewidth=2.5,
        label='Canary 2  (AORT7 gene • 52.7 / 58.4 mm)')

# ── Value annotations ─────────────────────────────────────────────────────────
for xi, (p1, n1, p2, n2) in enumerate(zip(c1_pct, c1_n, c2_pct, c2_n)):
    if p1 > 0:
        ax.annotate(f'{n1}/20', (xi, p1), textcoords='offset points',
                    xytext=(-22, 16), fontsize=8.5, color='#1565C0', fontweight='bold')
    if p2 > 0:
        ax.annotate(f'{n2}/20', (xi, p2), textcoords='offset points',
                    xytext=(-5, -18), fontsize=8.5, color='#B71C1C', fontweight='bold')

# ── Threshold marker ─────────────────────────────────────────────────────────
ax.axvline(x=2.5, color='#37474F', linestyle=':', linewidth=1.6, alpha=0.7)
ax.text(2.55, 88, 'LoRA threshold\n(3\u00d7 exposure)', fontsize=8.5,
        color='#37474F', va='top', style='italic')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_xlabel('Canary Injection Frequency', fontsize=13, labelpad=8)
ax.set_ylabel('Exact Size Extraction Rate\n(% of 20 generations)', fontsize=12, labelpad=8)
ax.set_title('Canary Dose–Response Curve: LoRA Parameter Memorization Threshold',
             fontsize=13, fontweight='bold', pad=14)
ax.set_ylim(-8, 115)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.legend(fontsize=10, loc='upper left', framealpha=0.85, edgecolor='#CCCCCC')
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines[['top','right']].set_visible(False)

# ── Zone labels ───────────────────────────────────────────────────────────────
ax.text(1.0, -6, 'No memorization (0×–2×)', fontsize=8, color='#607D8B',
        ha='center', style='italic')
ax.text(3.0, -6, 'Threshold zone', fontsize=8, color='#FF6F00',
        ha='center', style='italic')

plt.tight_layout()
out = 'data/results/figures/canary_dose_response.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved → {out}')
