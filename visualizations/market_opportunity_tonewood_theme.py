#!/usr/bin/env python3
"""
Starwood GuitarFX Market Opportunity Visualizations
TONEWOOD & MUSIC INSPIRED THEME
Featuring warm wood grain colors, acoustic guitar aesthetics, and organic feel.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory
os.makedirs('/home/ubuntu/Starwood/visualizations/output_tonewood', exist_ok=True)

# =============================================================================
# TONEWOOD & MUSIC INSPIRED COLOR PALETTE
# =============================================================================
# Inspired by premium guitar tonewoods and acoustic aesthetics

# Background colors - warm cream like aged spruce top
BG_CREAM = '#FDF5E6'  # Old lace - aged spruce
BG_WARM = '#FAF0E6'   # Linen - light wood

# Primary tonewoods
BRAZILIAN_ROSEWOOD = '#4A2C2A'  # Deep reddish-brown
COCOBOLO = '#8B4513'            # Saddle brown - orange-brown
HONDURAN_MAHOGANY = '#6B3A2E'   # Rich reddish-brown
KOA = '#B8860B'                 # Dark goldenrod - golden brown
EBONY = '#2C2416'               # Near black with brown undertone
MAPLE = '#F5DEB3'               # Wheat - light maple

# Accent colors - guitar hardware and strings
GOLD_HARDWARE = '#D4AF37'       # Metallic gold
NICKEL_STRINGS = '#C0C0C0'      # Silver
BRONZE_STRINGS = '#CD7F32'      # Bronze acoustic strings

# Text colors
TEXT_DARK = '#2C2416'           # Ebony for main text
TEXT_MEDIUM = '#5D4037'         # Brown for secondary text

# Create custom colormap for gradients
tonewood_colors = [BRAZILIAN_ROSEWOOD, COCOBOLO, KOA, MAPLE]

# Apply a clean style first
plt.style.use('seaborn-v0_8-whitegrid')

# Set custom styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'axes.facecolor': BG_CREAM,
    'figure.facecolor': BG_WARM,
    'text.color': TEXT_DARK,
    'axes.labelcolor': TEXT_DARK,
    'xtick.color': TEXT_MEDIUM,
    'ytick.color': TEXT_MEDIUM,
    'axes.edgecolor': COCOBOLO,
    'grid.color': '#DEB887',
    'grid.alpha': 0.4,
    'axes.linewidth': 2,
})

# =============================================================================
# Chart 1: Market Size & Growth Projection (2024-2030)
# =============================================================================
def create_market_growth_chart():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
    market_size = [1.2, 1.27, 1.35, 1.43, 1.52, 1.61, 1.70]
    ai_segment = [0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.24]
    
    bar_width = 0.35
    x = np.arange(len(years))
    
    # Total market bars - Brazilian Rosewood color
    bars1 = ax.bar(x - bar_width/2, market_size, bar_width, 
                   color=BRAZILIAN_ROSEWOOD, edgecolor=EBONY, linewidth=2,
                   label='Total Guitar Effects Market')
    
    # AI/Neural segment bars - Koa color
    bars2 = ax.bar(x + bar_width/2, ai_segment, bar_width,
                   color=KOA, edgecolor=COCOBOLO, linewidth=2,
                   label='AI/Neural Effects Segment')
    
    # Add value labels
    for bar, val in zip(bars1, market_size):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'${val:.2f}B', ha='center', va='bottom', fontsize=10,
                color=BRAZILIAN_ROSEWOOD, fontweight='bold')
    
    for bar, val in zip(bars2, ai_segment):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'${val*1000:.0f}M', ha='center', va='bottom', fontsize=9,
                color=COCOBOLO, fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_ylabel('Market Size (Billions USD)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('🎸 Guitar Effects Market Growth Projection (2024-2030)', 
                 fontsize=18, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper left', facecolor=BG_CREAM, edgecolor=COCOBOLO, fontsize=10)
    
    # Growth annotations with music note style
    ax.annotate('♪ 6% CAGR', xy=(5.5, 1.55), fontsize=12, color=BRAZILIAN_ROSEWOOD, fontweight='bold')
    ax.annotate('♫ 12% CAGR', xy=(5.5, 0.22), fontsize=12, color=KOA, fontweight='bold')
    
    # Add decorative wood grain border effect
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/01_market_growth_projection.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 01_market_growth_projection.png")

# =============================================================================
# Chart 2: Market Segments Breakdown (Donut Chart)
# =============================================================================
def create_market_segments_pie():
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG_WARM)
    
    segments = ['Distortion/Overdrive', 'Multi-FX Units', 'Delay/Reverb', 
                'Modulation', 'AI/Neural Effects', 'Acoustic Effects', 'Other']
    sizes = [320, 280, 240, 180, 120, 60, 50]
    
    # Tonewood-inspired colors
    colors = [BRAZILIAN_ROSEWOOD, HONDURAN_MAHOGANY, COCOBOLO, 
              '#8B6914', KOA, GOLD_HARDWARE, '#A0522D']
    explode = (0, 0, 0, 0, 0.12, 0.08, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=segments, 
                                       colors=colors, autopct='%1.1f%%',
                                       startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(edgecolor=BG_CREAM, linewidth=3))
    
    for text in texts:
        text.set_color(TEXT_DARK)
        text.set_fontsize(11)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color(BG_CREAM)
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Center circle - like a guitar soundhole
    centre_circle = plt.Circle((0, 0), 0.45, fc=BG_CREAM, ec=COCOBOLO, linewidth=4)
    ax.add_patch(centre_circle)
    
    # Soundhole rosette effect
    inner_ring = plt.Circle((0, 0), 0.42, fc='none', ec=BRAZILIAN_ROSEWOOD, linewidth=2)
    ax.add_patch(inner_ring)
    
    ax.text(0, 0.08, '$1.2B', ha='center', va='center', fontsize=32, 
            fontweight='bold', color=BRAZILIAN_ROSEWOOD)
    ax.text(0, -0.1, 'Total Market\n(2024)', ha='center', va='center', 
            fontsize=12, color=TEXT_MEDIUM)
    
    ax.set_title('🎵 Guitar Effects Market Segments (2024)', 
                 fontsize=18, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/02_market_segments_pie.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 02_market_segments_pie.png")

# =============================================================================
# Chart 3: Segment Growth Rates Comparison
# =============================================================================
def create_segment_growth_comparison():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    segments = ['AI/Neural\nEffects', 'Acoustic\nEffects', 'Multi-FX\nUnits', 
                'Delay/\nReverb', 'Modulation', 'Distortion/\nOverdrive']
    growth_rates = [12.0, 8.5, 7.2, 5.8, 4.5, 3.2]
    
    # Gradient colors based on growth rate - higher = more golden (like Koa)
    colors = [KOA if rate >= 8 else COCOBOLO if rate >= 6 else HONDURAN_MAHOGANY for rate in growth_rates]
    
    bars = ax.barh(segments, growth_rates, color=colors, edgecolor=EBONY, linewidth=2, height=0.6)
    
    # Add value labels with music notes
    for bar, rate in zip(bars, growth_rates):
        symbol = '♫' if rate >= 8 else '♪'
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{symbol} {rate}%', ha='left', va='center', fontsize=12,
                color=KOA if rate >= 8 else TEXT_DARK, fontweight='bold')
    
    # Average line - like a guitar fret
    avg_growth = 6.0
    ax.axvline(x=avg_growth, color=BRONZE_STRINGS, linestyle='-', linewidth=3, alpha=0.8)
    ax.text(avg_growth + 0.2, len(segments) - 0.3, f'Market Avg: {avg_growth}%', 
            color=BRONZE_STRINGS, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Annual Growth Rate (%)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('🎶 Growth Rate by Market Segment (CAGR 2024-2030)', 
                 fontsize=18, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    ax.set_xlim(0, 15)
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/03_segment_growth_rates.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 03_segment_growth_rates.png")

# =============================================================================
# Chart 4: Target Addressable Market (TAM/SAM/SOM) - Guitar Soundhole Style
# =============================================================================
def create_tam_sam_som():
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    # Concentric circles like guitar soundhole rosette
    circle_tam = plt.Circle((0.5, 0.5), 0.42, color=BRAZILIAN_ROSEWOOD, alpha=0.7)
    circle_sam = plt.Circle((0.5, 0.5), 0.28, color=COCOBOLO, alpha=0.8)
    circle_som = plt.Circle((0.5, 0.5), 0.14, color=KOA, alpha=0.9)
    
    # Decorative rings like rosette
    ring1 = plt.Circle((0.5, 0.5), 0.44, fc='none', ec=EBONY, linewidth=3)
    ring2 = plt.Circle((0.5, 0.5), 0.40, fc='none', ec=GOLD_HARDWARE, linewidth=2)
    ring3 = plt.Circle((0.5, 0.5), 0.30, fc='none', ec=GOLD_HARDWARE, linewidth=1.5)
    
    ax.add_patch(circle_tam)
    ax.add_patch(circle_sam)
    ax.add_patch(circle_som)
    ax.add_patch(ring1)
    ax.add_patch(ring2)
    ax.add_patch(ring3)
    
    # Labels
    ax.text(0.5, 0.88, 'TAM: $1.2B', ha='center', va='center', fontsize=16, 
            color=BG_CREAM, fontweight='bold')
    ax.text(0.5, 0.83, 'Total Guitar Effects Market', ha='center', va='center', 
            fontsize=11, color=MAPLE)
    
    ax.text(0.5, 0.70, 'SAM: $180M', ha='center', va='center', fontsize=14, 
            color=BG_CREAM, fontweight='bold')
    ax.text(0.5, 0.66, 'AI/Neural + Acoustic', ha='center', va='center', 
            fontsize=10, color=MAPLE)
    
    ax.text(0.5, 0.52, 'SOM', ha='center', va='center', fontsize=18, 
            color=EBONY, fontweight='bold')
    ax.text(0.5, 0.46, '$36M', ha='center', va='center', fontsize=16, 
            color=EBONY, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('🎸 Starwood GuitarFX Addressable Market', 
                 fontsize=18, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    
    # Legend with tonewood names
    legend_elements = [
        mpatches.Patch(facecolor=BRAZILIAN_ROSEWOOD, alpha=0.7, edgecolor=EBONY, 
                      label='TAM - Like Brazilian Rosewood: The Full Market'),
        mpatches.Patch(facecolor=COCOBOLO, alpha=0.8, edgecolor=EBONY,
                      label='SAM - Like Cocobolo: Our Target Segment'),
        mpatches.Patch(facecolor=KOA, alpha=0.9, edgecolor=EBONY,
                      label='SOM - Like Koa: 5-Year Realistic Capture')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
              facecolor=BG_CREAM, edgecolor=COCOBOLO, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/04_tam_sam_som.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 04_tam_sam_som.png")

# =============================================================================
# Chart 5: Revenue Projection (5-Year) - Stacked like guitar body layers
# =============================================================================
def create_revenue_projection():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    
    plugin_revenue = [0.5, 1.8, 3.5, 5.2, 7.0]
    mobile_revenue = [0.2, 0.8, 1.5, 2.5, 3.5]
    hardware_revenue = [0, 0.5, 3.0, 8.0, 15.0]
    
    x = np.arange(len(years))
    width = 0.55
    
    # Stacked bars - like guitar body layers (back, sides, top)
    bars1 = ax.bar(x, plugin_revenue, width, label='FX Plugin ($149) - Spruce Top', 
                   color=MAPLE, edgecolor=COCOBOLO, linewidth=2)
    bars2 = ax.bar(x, mobile_revenue, width, bottom=plugin_revenue, 
                   label='FX Mobile ($49) - Mahogany Sides', 
                   color=HONDURAN_MAHOGANY, edgecolor=COCOBOLO, linewidth=2)
    bars3 = ax.bar(x, hardware_revenue, width, 
                   bottom=np.array(plugin_revenue) + np.array(mobile_revenue), 
                   label='FX Hardware ($299) - Rosewood Back', 
                   color=BRAZILIAN_ROSEWOOD, edgecolor=EBONY, linewidth=2)
    
    # Total revenue line - like guitar strings
    total_revenue = [p + m + h for p, m, h in zip(plugin_revenue, mobile_revenue, hardware_revenue)]
    ax.plot(x, total_revenue, 'o-', color=GOLD_HARDWARE, linewidth=3, markersize=10, 
            label='Total Revenue', markeredgecolor=EBONY, markeredgewidth=2)
    
    for i, total in enumerate(total_revenue):
        ax.text(i, total + 0.8, f'${total:.1f}M', ha='center', va='bottom',
                fontsize=12, color=BRAZILIAN_ROSEWOOD, fontweight='bold')
    
    ax.set_xlabel('Timeline', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_ylabel('Revenue (Millions USD)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('🎵 Starwood GuitarFX 5-Year Revenue Projection', 
                 fontsize=18, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper left', facecolor=BG_CREAM, edgecolor=COCOBOLO, fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/05_revenue_projection.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 05_revenue_projection.png")

# =============================================================================
# Chart 6: Competitive Landscape - Fretboard Style
# =============================================================================
def create_competitive_landscape():
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    competitors = {
        'TonewoodAmp2': (249, 40, 250),
        'Neural Amp Modeler': (0, 70, 180),
        'Quad Cortex': (1599, 90, 350),
        'Line 6 Helix': (1499, 85, 300),
        'Boss GT-1000': (999, 75, 280),
        'Starwood GuitarFX': (199, 95, 450),
    }
    
    for name, (price, features, size) in competitors.items():
        if name == 'Starwood GuitarFX':
            ax.scatter(price, features, s=size, c=KOA, edgecolors=EBONY, 
                      linewidths=4, alpha=0.95, zorder=5, marker='*')
            ax.annotate(f'★ {name}', (price, features), xytext=(price + 100, features + 2),
                       fontsize=13, color=BRAZILIAN_ROSEWOOD, fontweight='bold')
        else:
            ax.scatter(price, features, s=size, c=COCOBOLO, edgecolors=HONDURAN_MAHOGANY, 
                      linewidths=2, alpha=0.7)
            ax.annotate(name, (price, features), xytext=(price + 50, features - 5),
                       fontsize=10, color=TEXT_MEDIUM)
    
    # Quadrant lines - like guitar frets
    ax.axhline(y=60, color=BRONZE_STRINGS, linestyle='-', alpha=0.5, linewidth=2)
    ax.axvline(x=500, color=BRONZE_STRINGS, linestyle='-', alpha=0.5, linewidth=2)
    
    # Quadrant labels
    ax.text(100, 98, '🎸 HIGH VALUE\n(Premium Tone, Fair Price)', fontsize=10, 
            color=KOA, ha='center', fontweight='bold')
    ax.text(1200, 98, '💎 PREMIUM\n(Premium Tone, Premium Price)', fontsize=10, 
            color=TEXT_MEDIUM, ha='center')
    ax.text(100, 25, '🔧 BUDGET\n(Basic Features)', fontsize=10, 
            color=TEXT_MEDIUM, ha='center')
    ax.text(1200, 25, '⚠️ OVERPRICED\n(Low Value)', fontsize=10, 
            color='#8B0000', ha='center')
    
    ax.set_xlabel('Price (USD)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_ylabel('Feature Richness Score', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('🎶 Competitive Landscape: Price vs. Features', 
                 fontsize=18, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    ax.set_xlim(-100, 1800)
    ax.set_ylim(0, 105)
    
    # Note
    ax.text(800, 8, '★ Starwood: Premium tonewood character at an accessible price',
            fontsize=12, color=KOA, style='italic', fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/06_competitive_landscape.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 06_competitive_landscape.png")

# =============================================================================
# Run all chart generators
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("🎸 Generating Starwood GuitarFX Market Visualizations")
    print("   TONEWOOD & MUSIC INSPIRED THEME")
    print("=" * 70)
    
    create_market_growth_chart()
    create_market_segments_pie()
    create_segment_growth_comparison()
    create_tam_sam_som()
    create_revenue_projection()
    create_competitive_landscape()
    
    print("=" * 70)
    print("✅ All visualizations generated successfully!")
    print("📁 Output: /home/ubuntu/Starwood/visualizations/output_tonewood/")
    print("=" * 70)
