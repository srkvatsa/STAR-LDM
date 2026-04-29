"""Generate STAR-LDM architecture and diffusion loop diagrams for poster."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ─── Architecture / Diffusion Loop Diagram ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.axis('off')

def box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='white' if color in ['#B71C1C','#1565C0','#2E7D32','#4A148C','#E65100'] else '#1a1a1a')

def arrow(ax, x1, y1, x2, y2, color='#555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

def label(ax, x, y, text, fontsize=9, color='#555'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=color, fontstyle='italic')

# Title
ax.text(5, 6.2, 'STAR-LDM Inference: One Diffusion Step', ha='center', fontsize=14, fontweight='bold')

# Input
box(ax, 0.1, 3.5, 1.4, 0.9, 'Noised\nEmbedding\n$z_t$ (768)', '#4A148C', fontsize=9, bold=True)
label(ax, 0.8, 3.0, 'from previous\nstep or N(0,I)', fontsize=8)

# Noise level
box(ax, 0.1, 5.0, 1.4, 0.7, 'Noise Level\n$\\alpha^2(t)$', '#E65100', fontsize=9, bold=True)

# SPG
box(ax, 2.2, 3.2, 1.8, 1.4, 'Soft Prompt\nGenerator\n6 layers, dim 1024\nnon-causal', '#1565C0', fontsize=9, bold=True)
arrow(ax, 1.5, 3.95, 2.2, 3.95)
label(ax, 1.85, 4.2, '(B,768)', fontsize=8)

# Time embedding arrow to SPG
arrow(ax, 1.5, 5.3, 2.2, 4.3, color='#E65100')
label(ax, 1.6, 4.9, 'FiLM\ncond.', fontsize=7, color='#E65100')

# SPG output
arrow(ax, 4.0, 3.95, 4.7, 3.95)
label(ax, 4.35, 4.2, '8 soft prompts\n(B,8,1280)', fontsize=8)

# GPT-2
box(ax, 4.7, 2.8, 2.0, 2.2, 'GPT-2 Large\n36 layers\n770M params\n\n+ prefix KV cache', '#B71C1C', fontsize=9, bold=True)

# Prefix KV
box(ax, 4.9, 1.5, 1.6, 0.8, 'Cached Prefix\nKV (computed once)', '#FFCDD2', fontsize=8)
arrow(ax, 5.7, 2.3, 5.7, 2.8, color='#B71C1C')
label(ax, 6.7, 2.0, 'only 8 tokens\nthrough GPT-2\n(not full prefix!)', fontsize=8, color='#B71C1C')

# GPT-2 output
arrow(ax, 6.7, 3.95, 7.3, 3.95)
label(ax, 7.0, 4.2, 'hidden states\n(B,8,1280)', fontsize=8)

# Concat
box(ax, 7.3, 3.5, 0.6, 0.9, 'cat', '#FFE0B2', fontsize=9)
# soft prompt also goes to concat
ax.annotate('', xy=(7.6, 4.4), xytext=(4.0, 4.6),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5, connectionstyle='arc3,rad=0.3'))
label(ax, 5.5, 5.1, 'soft prompts skip connection', fontsize=7, color='#1565C0')

arrow(ax, 7.9, 3.95, 8.2, 3.95)
label(ax, 8.05, 4.2, '(B,8,2560)', fontsize=8)

# ScoreNet
box(ax, 8.2, 3.2, 1.6, 1.4, 'Score Net\nHead\n6 layers\ndim 1024', '#2E7D32', fontsize=9, bold=True)
# Time emb to ScoreNet
arrow(ax, 1.5, 5.35, 9.0, 5.35, color='#E65100')
ax.annotate('', xy=(9.0, 4.6), xytext=(9.0, 5.35),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))

# Output
box(ax, 8.4, 1.5, 1.2, 0.9, 'v-prediction\n$\\hat{v}$ (768)', '#4A148C', fontsize=9, bold=True)
arrow(ax, 9.0, 3.2, 9.0, 2.4)

# DDPM step
box(ax, 5.5, 0.3, 2.2, 0.8, 'DDPM Update\n$z_{t-1} = f(z_t, \\hat{v}, \\alpha^2)$', '#FF6F00', fontsize=9, bold=True)
arrow(ax, 8.4, 1.9, 7.7, 1.1, color='#4A148C')
arrow(ax, 5.5, 0.7, 0.8, 0.7, color='#4A148C')
ax.annotate('', xy=(0.8, 3.5), xytext=(0.8, 1.1),
            arrowprops=dict(arrowstyle='->', color='#4A148C', lw=2, connectionstyle='arc3,rad=0'))
label(ax, 3.0, 0.4, '$z_{t-1}$ feeds back for next step (repeat $T$ times)', fontsize=9, color='#333')

# Legend for what we optimized
ax.text(0.1, 6.2, 'Red outline = optimization target (77% of runtime)', fontsize=9, color='#B71C1C')

plt.tight_layout()
plt.savefig('figures/fig0_architecture.pdf', bbox_inches='tight')
plt.savefig('figures/fig0_architecture.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig0_architecture")
