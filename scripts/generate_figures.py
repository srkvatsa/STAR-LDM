"""Generate all figures for the poster and paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Consistent styling
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
})
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#607D8B']

# ─── Figure 1: Profiling Pie Chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
labels = ['GPT-2 forward\n(diffusion loop)', 'GPT-2 generate\n(AR decode)',
          'SoftPrompt\nGenerator', 'ScoreNet\nHead', 'Other']
sizes = [38.7, 38.5, 9.1, 9.2, 4.5]
colors = ['#E53935', '#FF7043', '#42A5F5', '#66BB6A', '#BDBDBD']
explode = (0.05, 0.05, 0, 0, 0)
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                   colors=colors, explode=explode,
                                   startangle=90, pctdistance=0.75)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
ax.set_title('Baseline Runtime Breakdown\n(50 steps, 50 C4 prompts, unoptimized)')
# Add annotation for GPT-2 total
fig.text(0.5, 0.02, 'GPT-2 backbone accounts for 77.2% of total inference time',
         ha='center', fontsize=12, fontstyle='italic')
plt.tight_layout()
plt.savefig('figures/fig1_profiling_pie.pdf', bbox_inches='tight')
plt.savefig('figures/fig1_profiling_pie.png', bbox_inches='tight')
plt.close()
print("Saved fig1_profiling_pie")

# ─── Figure 2: Prefix Length Scaling ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

prefix_lens = [16, 64, 128, 256, 512]
baseline =    [921, 1072, 1321, 1835, 2894]
optimized =   [748, 765, 801, 853, 982]
speedups =    [b/o for b, o in zip(baseline, optimized)]

ax1.plot(prefix_lens, baseline, 'o-', color=COLORS[3], linewidth=2.5, markersize=8, label='Baseline (HuggingFace)')
ax1.plot(prefix_lens, optimized, 's-', color=COLORS[2], linewidth=2.5, markersize=8, label='Optimized (this work)')
ax1.fill_between(prefix_lens, optimized, baseline, alpha=0.15, color=COLORS[2])
ax1.set_xlabel('Prefix Length (tokens)')
ax1.set_ylabel('End-to-End Latency (ms)')
ax1.set_title('Latency vs. Prefix Length (20 steps)')
ax1.legend(loc='upper left')
ax1.set_xlim(0, 540)
ax1.set_ylim(0, 3200)
ax1.grid(True, alpha=0.3)
# Annotate speedups
for x, b, o, s in zip(prefix_lens, baseline, optimized, speedups):
    ax1.annotate(f'{s:.1f}x', xy=(x, (b+o)/2), fontsize=10, ha='center',
                fontweight='bold', color=COLORS[0])

# Right: speedup curve
ax2.bar(range(len(prefix_lens)), speedups, color=COLORS[0], alpha=0.85, edgecolor='white', linewidth=1.5)
ax2.set_xticks(range(len(prefix_lens)))
ax2.set_xticklabels([str(p) for p in prefix_lens])
ax2.set_xlabel('Prefix Length (tokens)')
ax2.set_ylabel('Speedup (x)')
ax2.set_title('Speedup Scales with Context Length')
ax2.set_ylim(0, 3.5)
ax2.grid(True, alpha=0.3, axis='y')
for i, s in enumerate(speedups):
    ax2.text(i, s + 0.08, f'{s:.2f}x', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('figures/fig2_prefix_scaling.pdf', bbox_inches='tight')
plt.savefig('figures/fig2_prefix_scaling.png', bbox_inches='tight')
plt.close()
print("Saved fig2_prefix_scaling")

# ─── Figure 3: Roofline Plot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Hardware ceilings
peak_gflops = 14000
mem_bw = 546  # GB/s
ridge = peak_gflops / mem_bw  # ~25.6

ai_range = np.logspace(-2, 3, 500)
roofline = np.minimum(peak_gflops, ai_range * mem_bw)
ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='M4 Max Roofline')
ax.axvline(x=ridge, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Data points
points = [
    ('RMSNorm+FiLM\n(JIT)', 0.74, 2.1, COLORS[5]),
    ('RMSNorm+FiLM\n(Metal)', 0.74, 4.5, COLORS[2]),
    ('Tiny Attn\n(JIT)', 2.28, 3.2, COLORS[5]),
    ('Tiny Attn\n(Metal)', 2.28, 7.4, COLORS[2]),
    ('DDPM Step\n(JIT)', 1.25, 0.37, COLORS[5]),
    ('GPT-2\ndecode-8', 4.0, 120, COLORS[3]),
    ('SPG\n(6 layers)', 4.04, 440, COLORS[0]),
]

for name, ai, gflops, color in points:
    ax.plot(ai, gflops, 'o', color=color, markersize=12, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5)
    # Offset labels
    offset = (10, 5) if gflops > 10 else (10, -15)
    ax.annotate(name, xy=(ai, gflops), xytext=offset,
               textcoords='offset points', fontsize=9, ha='left')

# Shade the dispatch-latency-bound region
ax.axhspan(0.01, 10, alpha=0.08, color='red')
ax.text(0.015, 0.15, 'Dispatch-latency-bound\nregime (<1% ceiling)',
        fontsize=10, fontstyle='italic', color='#B71C1C')

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
ax.set_ylabel('Achieved Throughput (GFLOP/s)')
ax.set_title('Roofline Analysis: STAR-LDM Components on M4 Max')
ax.set_xlim(0.01, 1000)
ax.set_ylim(0.05, 20000)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.2, which='both')

# Add legend for colors
metal_patch = mpatches.Patch(color=COLORS[2], label='Metal kernel')
jit_patch = mpatches.Patch(color=COLORS[5], label='JIT / PyTorch')
model_patch = mpatches.Patch(color=COLORS[0], label='Full module')
gpt2_patch = mpatches.Patch(color=COLORS[3], label='GPT-2 backbone')
ax.legend(handles=[metal_patch, jit_patch, model_patch, gpt2_patch,
                   plt.Line2D([0],[0], color='k', linewidth=2.5, label='Roofline')],
         loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig3_roofline.pdf', bbox_inches='tight')
plt.savefig('figures/fig3_roofline.png', bbox_inches='tight')
plt.close()
print("Saved fig3_roofline")

# ─── Figure 4: Dispatch Count Comparison ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

categories = ['HuggingFace\nGPT-2 Forward', 'Streamlined\nForward (ours)']
counts = [6185, 432]
bars = ax.bar(categories, counts, color=[COLORS[3], COLORS[2]], width=0.5,
              edgecolor='white', linewidth=2)
ax.set_ylabel('ATen Operator Dispatches per Call')
ax.set_title('Framework Overhead: Operator Dispatch Count')

# Annotate
ax.text(0, 6185 + 150, '6,185', ha='center', fontsize=16, fontweight='bold', color=COLORS[3])
ax.text(1, 432 + 150, '432', ha='center', fontsize=16, fontweight='bold', color=COLORS[2])

# Arrow showing reduction
ax.annotate('93% reduction', xy=(0.5, 3300), fontsize=14, ha='center',
           fontweight='bold', color=COLORS[0],
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=COLORS[0]))

ax.set_ylim(0, 7500)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('figures/fig4_dispatch_count.pdf', bbox_inches='tight')
plt.savefig('figures/fig4_dispatch_count.png', bbox_inches='tight')
plt.close()
print("Saved fig4_dispatch_count")

# ─── Figure 5: Kernel Microbenchmarks ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

kernels = ['RMSNorm\n+FiLM', 'Tiny\nAttention', 'DDPM\nStep', 'Decode-N\nAttn (S=13)', 'Fused\nFFN']
jit_ms = [0.027, 0.099, 0.040, 0.021, 0.124]
metal_ms = [0.016, 0.042, 0.115, 0.011, 15.77]
speedups_k = [j/m for j, m in zip(jit_ms, metal_ms)]

x = np.arange(len(kernels))
w = 0.35
b1 = ax.bar(x - w/2, jit_ms, w, label='PyTorch JIT/SDPA', color=COLORS[5], edgecolor='white')
b2 = ax.bar(x + w/2, metal_ms, w, label='Metal Kernel', color=COLORS[2], edgecolor='white')

ax.set_ylabel('Latency (ms)')
ax.set_title('Metal Kernel Microbenchmarks')
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.legend()
ax.set_yscale('log')
ax.set_ylim(0.005, 50)
ax.grid(True, alpha=0.2, axis='y')

# Annotate speedups
for i, s in enumerate(speedups_k):
    color = COLORS[2] if s > 1 else COLORS[3]
    label = f'{s:.1f}x' if s >= 1 else f'{s:.2f}x'
    ax.text(i, max(jit_ms[i], metal_ms[i]) * 1.3, label,
           ha='center', fontsize=10, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('figures/fig5_kernel_microbench.pdf', bbox_inches='tight')
plt.savefig('figures/fig5_kernel_microbench.png', bbox_inches='tight')
plt.close()
print("Saved fig5_kernel_microbench")

print("\nAll figures saved to figures/")
