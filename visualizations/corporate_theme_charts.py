#!/usr/bin/env python3
"""
Starwood GuitarFX Market Opportunity & Sustainability Visualizations
CORPORATE PROFESSIONAL THEME
Clean blues, grays, and whites for investor presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Circle
import os

# Create output directory
os.makedirs('/home/ubuntu/Starwood/visualizations/output_corporate', exist_ok=True)

# =============================================================================
# CORPORATE PROFESSIONAL COLOR PALETTE
# =============================================================================
# Clean, professional colors for business presentations

# Backgrounds
BG_WHITE = '#FFFFFF'
BG_LIGHT_GRAY = '#F8F9FA'

# Primary corporate blues
CORP_NAVY = '#1A365D'        # Deep navy - primary
CORP_BLUE = '#2B6CB0'        # Medium blue - secondary
CORP_LIGHT_BLUE = '#4299E1'  # Light blue - accent
CORP_SKY = '#90CDF4'         # Sky blue - highlights

# Grays
CORP_DARK_GRAY = '#2D3748'   # Dark gray - text
CORP_MEDIUM_GRAY = '#718096' # Medium gray - secondary text
CORP_LIGHT_GRAY = '#E2E8F0'  # Light gray - borders

# Accent colors
CORP_GREEN = '#38A169'       # Success green
CORP_RED = '#E53E3E'         # Alert red
CORP_ORANGE = '#DD6B20'      # Warning orange
CORP_TEAL = '#319795'        # Teal accent

# Text
TEXT_PRIMARY = '#1A202C'
TEXT_SECONDARY = '#4A5568'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.facecolor': BG_WHITE,
    'figure.facecolor': BG_LIGHT_GRAY,
    'text.color': TEXT_PRIMARY,
    'axes.labelcolor': TEXT_PRIMARY,
    'xtick.color': TEXT_SECONDARY,
    'ytick.color': TEXT_SECONDARY,
    'axes.edgecolor': CORP_LIGHT_GRAY,
    'grid.color': CORP_LIGHT_GRAY,
    'grid.alpha': 0.7,
    'axes.linewidth': 1.5,
})

# =============================================================================
# Chart 1: Market Size & Growth Projection (2024-2030)
# =============================================================================
def create_market_growth_chart():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
    market_size = [1.2, 1.27, 1.35, 1.43, 1.52, 1.61, 1.70]
    ai_segment = [0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.24]
    
    bar_width = 0.35
    x = np.arange(len(years))
    
    bars1 = ax.bar(x - bar_width/2, market_size, bar_width, 
                   color=CORP_NAVY, edgecolor=CORP_DARK_GRAY, linewidth=1,
                   label='Total Guitar Effects Market')
    
    bars2 = ax.bar(x + bar_width/2, ai_segment, bar_width,
                   color=CORP_LIGHT_BLUE, edgecolor=CORP_BLUE, linewidth=1,
                   label='AI/Neural Effects Segment')
    
    for bar, val in zip(bars1, market_size):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'${val:.2f}B', ha='center', va='bottom', fontsize=10,
                color=CORP_NAVY, fontweight='bold')
    
    for bar, val in zip(bars2, ai_segment):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'${val*1000:.0f}M', ha='center', va='bottom', fontsize=9,
                color=CORP_BLUE, fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_ylabel('Market Size (Billions USD)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Guitar Effects Market Growth Projection (2024-2030)', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper left', facecolor=BG_WHITE, edgecolor=CORP_LIGHT_GRAY, fontsize=10)
    
    ax.annotate('6% CAGR', xy=(5.5, 1.55), fontsize=11, color=CORP_NAVY, fontweight='bold')
    ax.annotate('12% CAGR', xy=(5.5, 0.22), fontsize=11, color=CORP_LIGHT_BLUE, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/01_market_growth_projection.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 01_market_growth_projection.png")

# =============================================================================
# Chart 2: Market Segments Breakdown (Donut Chart)
# =============================================================================
def create_market_segments_pie():
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    
    segments = ['Distortion/Overdrive', 'Multi-FX Units', 'Delay/Reverb', 
                'Modulation', 'AI/Neural Effects', 'Acoustic Effects', 'Other']
    sizes = [320, 280, 240, 180, 120, 60, 50]
    
    colors = [CORP_NAVY, CORP_BLUE, CORP_LIGHT_BLUE, CORP_TEAL, 
              CORP_GREEN, CORP_ORANGE, CORP_MEDIUM_GRAY]
    explode = (0, 0, 0, 0, 0.1, 0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=segments, 
                                       colors=colors, autopct='%1.1f%%',
                                       startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(edgecolor=BG_WHITE, linewidth=2))
    
    for text in texts:
        text.set_color(TEXT_PRIMARY)
        text.set_fontsize(10)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color(BG_WHITE)
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.45, fc=BG_WHITE, ec=CORP_LIGHT_GRAY, linewidth=2)
    ax.add_patch(centre_circle)
    
    ax.text(0, 0.05, '$1.2B', ha='center', va='center', fontsize=28, 
            fontweight='bold', color=CORP_NAVY)
    ax.text(0, -0.1, 'Total Market\n(2024)', ha='center', va='center', 
            fontsize=11, color=TEXT_SECONDARY)
    
    ax.set_title('Guitar Effects Market Segments (2024)', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/02_market_segments_pie.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 02_market_segments_pie.png")

# =============================================================================
# Chart 3: Segment Growth Rates Comparison
# =============================================================================
def create_segment_growth_comparison():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    segments = ['AI/Neural\nEffects', 'Acoustic\nEffects', 'Multi-FX\nUnits', 
                'Delay/\nReverb', 'Modulation', 'Distortion/\nOverdrive']
    growth_rates = [12.0, 8.5, 7.2, 5.8, 4.5, 3.2]
    
    colors = [CORP_GREEN if rate >= 8 else CORP_BLUE if rate >= 6 else CORP_LIGHT_BLUE for rate in growth_rates]
    
    bars = ax.barh(segments, growth_rates, color=colors, edgecolor=CORP_DARK_GRAY, linewidth=1, height=0.6)
    
    for bar, rate in zip(bars, growth_rates):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{rate}%', ha='left', va='center', fontsize=11,
                color=CORP_GREEN if rate >= 8 else TEXT_PRIMARY, fontweight='bold')
    
    avg_growth = 6.0
    ax.axvline(x=avg_growth, color=CORP_ORANGE, linestyle='--', linewidth=2, alpha=0.8)
    ax.text(avg_growth + 0.2, len(segments) - 0.3, f'Market Avg: {avg_growth}%', 
            color=CORP_ORANGE, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Annual Growth Rate (%)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Growth Rate by Market Segment (CAGR 2024-2030)', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_xlim(0, 15)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/03_segment_growth_rates.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 03_segment_growth_rates.png")

# =============================================================================
# Chart 4: Target Addressable Market (TAM/SAM/SOM)
# =============================================================================
def create_tam_sam_som():
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    circle_tam = plt.Circle((0.5, 0.5), 0.42, color=CORP_NAVY, alpha=0.6)
    circle_sam = plt.Circle((0.5, 0.5), 0.28, color=CORP_BLUE, alpha=0.7)
    circle_som = plt.Circle((0.5, 0.5), 0.14, color=CORP_GREEN, alpha=0.9)
    
    ring1 = plt.Circle((0.5, 0.5), 0.44, fc='none', ec=CORP_DARK_GRAY, linewidth=2)
    
    ax.add_patch(circle_tam)
    ax.add_patch(circle_sam)
    ax.add_patch(circle_som)
    ax.add_patch(ring1)
    
    ax.text(0.5, 0.88, 'TAM: $1.2B', ha='center', va='center', fontsize=14, 
            color=BG_WHITE, fontweight='bold')
    ax.text(0.5, 0.83, 'Total Guitar Effects Market', ha='center', va='center', 
            fontsize=10, color=CORP_SKY)
    
    ax.text(0.5, 0.70, 'SAM: $180M', ha='center', va='center', fontsize=13, 
            color=BG_WHITE, fontweight='bold')
    ax.text(0.5, 0.66, 'AI/Neural + Acoustic', ha='center', va='center', 
            fontsize=9, color=CORP_SKY)
    
    ax.text(0.5, 0.52, 'SOM', ha='center', va='center', fontsize=16, 
            color=BG_WHITE, fontweight='bold')
    ax.text(0.5, 0.46, '$36M', ha='center', va='center', fontsize=14, 
            color=BG_WHITE, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('Starwood GuitarFX Addressable Market', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    
    legend_elements = [
        mpatches.Patch(facecolor=CORP_NAVY, alpha=0.6, edgecolor=CORP_DARK_GRAY, 
                      label='TAM - Total Addressable Market'),
        mpatches.Patch(facecolor=CORP_BLUE, alpha=0.7, edgecolor=CORP_DARK_GRAY,
                      label='SAM - Serviceable Addressable Market'),
        mpatches.Patch(facecolor=CORP_GREEN, alpha=0.9, edgecolor=CORP_DARK_GRAY,
                      label='SOM - Serviceable Obtainable Market')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
              facecolor=BG_WHITE, edgecolor=CORP_LIGHT_GRAY, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/04_tam_sam_som.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 04_tam_sam_som.png")

# =============================================================================
# Chart 5: Revenue Projection (5-Year)
# =============================================================================
def create_revenue_projection():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    
    plugin_revenue = [0.5, 1.8, 3.5, 5.2, 7.0]
    mobile_revenue = [0.2, 0.8, 1.5, 2.5, 3.5]
    hardware_revenue = [0, 0.5, 3.0, 8.0, 15.0]
    
    x = np.arange(len(years))
    width = 0.55
    
    bars1 = ax.bar(x, plugin_revenue, width, label='FX Plugin ($149)', 
                   color=CORP_LIGHT_BLUE, edgecolor=CORP_BLUE, linewidth=1)
    bars2 = ax.bar(x, mobile_revenue, width, bottom=plugin_revenue, 
                   label='FX Mobile ($49)', 
                   color=CORP_BLUE, edgecolor=CORP_NAVY, linewidth=1)
    bars3 = ax.bar(x, hardware_revenue, width, 
                   bottom=np.array(plugin_revenue) + np.array(mobile_revenue), 
                   label='FX Hardware ($299)', 
                   color=CORP_NAVY, edgecolor=CORP_DARK_GRAY, linewidth=1)
    
    total_revenue = [p + m + h for p, m, h in zip(plugin_revenue, mobile_revenue, hardware_revenue)]
    ax.plot(x, total_revenue, 'o-', color=CORP_GREEN, linewidth=3, markersize=10, 
            label='Total Revenue', markeredgecolor=CORP_DARK_GRAY, markeredgewidth=1)
    
    for i, total in enumerate(total_revenue):
        ax.text(i, total + 0.8, f'${total:.1f}M', ha='center', va='bottom',
                fontsize=11, color=CORP_NAVY, fontweight='bold')
    
    ax.set_xlabel('Timeline', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_ylabel('Revenue (Millions USD)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Starwood GuitarFX 5-Year Revenue Projection', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper left', facecolor=BG_WHITE, edgecolor=CORP_LIGHT_GRAY, fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/05_revenue_projection.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 05_revenue_projection.png")

# =============================================================================
# Chart 6: Competitive Landscape
# =============================================================================
def create_competitive_landscape():
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
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
            ax.scatter(price, features, s=size, c=CORP_GREEN, edgecolors=CORP_DARK_GRAY, 
                      linewidths=3, alpha=0.9, zorder=5, marker='*')
            ax.annotate(f'* {name}', (price, features), xytext=(price + 100, features + 2),
                       fontsize=12, color=CORP_GREEN, fontweight='bold')
        else:
            ax.scatter(price, features, s=size, c=CORP_BLUE, edgecolors=CORP_NAVY, 
                      linewidths=1.5, alpha=0.7)
            ax.annotate(name, (price, features), xytext=(price + 50, features - 5),
                       fontsize=10, color=TEXT_SECONDARY)
    
    ax.axhline(y=60, color=CORP_LIGHT_GRAY, linestyle='--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=500, color=CORP_LIGHT_GRAY, linestyle='--', alpha=0.8, linewidth=1.5)
    
    ax.text(100, 98, 'HIGH VALUE', fontsize=10, color=CORP_GREEN, ha='center', fontweight='bold')
    ax.text(1200, 98, 'PREMIUM', fontsize=10, color=TEXT_SECONDARY, ha='center')
    ax.text(100, 25, 'BUDGET', fontsize=10, color=TEXT_SECONDARY, ha='center')
    ax.text(1200, 25, 'OVERPRICED', fontsize=10, color=CORP_RED, ha='center')
    
    ax.set_xlabel('Price (USD)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_ylabel('Feature Richness Score', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Competitive Landscape: Price vs. Features', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_xlim(-100, 1800)
    ax.set_ylim(0, 105)
    
    ax.text(800, 8, '* Starwood offers the highest feature score at an accessible price point',
            fontsize=11, color=CORP_GREEN, style='italic', fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/06_competitive_landscape.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 06_competitive_landscape.png")

# =============================================================================
# Chart 7: Conservation Impact
# =============================================================================
def create_conservation_impact():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    adoption_rates = ['5%', '10%', '25%', '50%', '75%']
    trees_saved = [2500, 5000, 12500, 25000, 37500]
    
    colors = [CORP_GREEN] * 5
    
    bars = ax.bar(adoption_rates, trees_saved, color=colors, edgecolor=CORP_DARK_GRAY, linewidth=1, width=0.6)
    
    for bar, trees in zip(bars, trees_saved):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
                f'{trees:,}\nTrees', ha='center', va='bottom', fontsize=10,
                color=CORP_GREEN, fontweight='bold')
    
    ax.set_xlabel('Starwood Adoption Rate Among Guitarists', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_ylabel('Tonewood Trees Saved Per Year', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Conservation Impact: Trees Saved Through Neural Tonewood Technology', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_ylim(0, 45000)
    
    ax.text(0.5, 0.92, 'Brazilian Rosewood: CITES Appendix I Protected (Critically Endangered)',
            transform=ax.transAxes, fontsize=10, color=CORP_RED, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FED7D7', edgecolor=CORP_RED, alpha=0.8))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/07_conservation_impact.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 07_conservation_impact.png")

# =============================================================================
# Chart 8: Environmental Comparison
# =============================================================================
def create_environmental_comparison():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    categories = ['Trees\nHarvested', 'CO2\nFootprint\n(kg)', 'Water\nUsage\n(liters)', 
                  'Habitat\nDestruction\n(sq m)', 'Years to\nRegrow']
    
    traditional = [3, 150, 2000, 50, 100]
    starwood = [0, 5, 10, 0, 0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, traditional, width, label='Traditional Premium Guitar', 
                   color=CORP_RED, edgecolor=CORP_DARK_GRAY, linewidth=1, alpha=0.8)
    bars2 = ax.bar(x + width/2, starwood, width, label='Starwood GuitarFX', 
                   color=CORP_GREEN, edgecolor=CORP_DARK_GRAY, linewidth=1, alpha=0.8)
    
    for bar, val in zip(bars1, traditional):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val}', ha='center', va='bottom', fontsize=10,
                    color=CORP_RED, fontweight='bold')
    
    for bar, val in zip(bars2, starwood):
        label = 'ZERO' if val == 0 else str(val)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha='center', va='bottom', fontsize=10,
                color=CORP_GREEN, fontweight='bold')
    
    ax.set_ylabel('Environmental Impact (Various Units)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Environmental Footprint: Traditional Guitar vs. Starwood', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', facecolor=BG_WHITE, edgecolor=CORP_LIGHT_GRAY, fontsize=10)
    ax.set_ylim(0, 180)
    
    ax.text(0.5, 0.88, '97% REDUCTION in Environmental Impact',
            transform=ax.transAxes, fontsize=13, color=CORP_GREEN, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#C6F6D5', edgecolor=CORP_GREEN, alpha=0.9))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/08_environmental_comparison.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 08_environmental_comparison.png")

# =============================================================================
# Chart 9: CITES Protected Tonewoods Status
# =============================================================================
def create_cites_status():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    tonewoods = ['Brazilian\nRosewood', 'Cocobolo', 'Honduran\nMahogany', 
                 'African\nBlackwood', 'Ebony', 'Bubinga']
    
    cites_level = [1, 2, 2, 2, 2, 2]
    price_premium = [15000, 3000, 1500, 2500, 2000, 1800]
    
    colors = [CORP_RED if level == 1 else CORP_ORANGE for level in cites_level]
    
    bars = ax.barh(tonewoods, price_premium, color=colors, edgecolor=CORP_DARK_GRAY, linewidth=1, height=0.6)
    
    cites_labels = ['CITES I\nCRITICAL', 'CITES II\nTHREATENED', 'CITES II\nTHREATENED',
                    'CITES II\nTHREATENED', 'CITES II\nTHREATENED', 'CITES II\nTHREATENED']
    
    for bar, label, premium in zip(bars, cites_labels, price_premium):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f'${premium:,}', ha='left', va='center', fontsize=10,
                color=TEXT_PRIMARY, fontweight='bold')
        ax.text(500, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=8,
                color='white', fontweight='bold')
    
    ax.set_xlabel('Price Premium for Traditional Guitar ($)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('CITES Protected Tonewoods: Endangered Species & Price Premiums', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_xlim(0, 18000)
    
    ax.text(0.65, 0.15, 'STARWOOD SOLUTION:\nAll premium tones\nZERO endangered wood\n$149 - $299',
            transform=ax.transAxes, fontsize=11, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=CORP_GREEN, edgecolor=CORP_DARK_GRAY, alpha=0.95))
    
    legend_elements = [
        mpatches.Patch(facecolor=CORP_RED, edgecolor=CORP_DARK_GRAY, label='CITES Appendix I - Critically Endangered'),
        mpatches.Patch(facecolor=CORP_ORANGE, edgecolor=CORP_DARK_GRAY, label='CITES Appendix II - Threatened'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', facecolor=BG_WHITE, edgecolor=CORP_LIGHT_GRAY)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/09_cites_status.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 09_cites_status.png")

# =============================================================================
# Chart 10: Ethical Value Proposition Summary
# =============================================================================
def create_ethical_summary():
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    metrics = [
        ('ZERO', 'Endangered Trees Cut', CORP_GREEN),
        ('100%', 'Digital & Sustainable', CORP_BLUE),
        ('$14,850', 'Savings vs. Brazilian\nRosewood Guitar', CORP_TEAL),
        ('97%', 'Lower Carbon\nFootprint', CORP_LIGHT_BLUE),
        ('INFINITE', 'Tonewood Variety\nWithout Harm', CORP_NAVY),
    ]
    
    positions = [(0.15, 0.7), (0.5, 0.7), (0.85, 0.7), (0.3, 0.3), (0.7, 0.3)]
    
    for (value, label, color), (x, y) in zip(metrics, positions):
        circle = Circle((x, y), 0.12, fc=color, ec=CORP_DARK_GRAY, linewidth=2, transform=ax.transAxes)
        ax.add_patch(circle)
        
        ax.text(x, y + 0.02, value, transform=ax.transAxes, ha='center', va='center',
                fontsize=14 if len(value) < 5 else 11, fontweight='bold', color='white')
        
        ax.text(x, y - 0.18, label, transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold', color=TEXT_PRIMARY)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'STARWOOD ETHICAL VALUE PROPOSITION', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=18, fontweight='bold', color=CORP_NAVY)
    
    ax.text(0.5, 0.88, 'Premium Guitar Tone Without Environmental Cost', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color=TEXT_SECONDARY, style='italic')
    
    ax.text(0.5, 0.08, '"Every Starwood user helps preserve endangered forests\nwhile enjoying the world\'s finest guitar tones."',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, color=CORP_GREEN, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='#C6F6D5', edgecolor=CORP_GREEN, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/10_ethical_summary.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 10_ethical_summary.png")

# =============================================================================
# Chart 11: Deforestation Timeline & Starwood Impact
# =============================================================================
def create_deforestation_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_LIGHT_GRAY)
    ax.set_facecolor(BG_WHITE)
    
    years = [1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
    
    rosewood_decline = [100, 85, 60, 35, 15, 8, 5, 3]
    rosewood_with_starwood = [100, 85, 60, 35, 15, 8, 5, 7]
    
    ax.fill_between(years[:7], rosewood_decline[:7], alpha=0.3, color=CORP_RED)
    ax.plot(years[:7], rosewood_decline[:7], 'o-', color=CORP_RED, linewidth=3, 
            markersize=8, label='Historical Decline')
    
    ax.fill_between(years[5:], rosewood_with_starwood[5:], alpha=0.3, color=CORP_GREEN)
    ax.plot(years[5:], rosewood_with_starwood[5:], 'o--', color=CORP_GREEN, linewidth=3, 
            markersize=8, label='Projected with Starwood Adoption')
    
    ax.axvline(x=1992, color=CORP_DARK_GRAY, linestyle=':', linewidth=2, alpha=0.7)
    ax.text(1992, 80, 'CITES\nBan\n1992', ha='center', fontsize=9, color=CORP_DARK_GRAY, fontweight='bold')
    
    ax.axvline(x=2025, color=CORP_GREEN, linestyle=':', linewidth=2, alpha=0.7)
    ax.text(2025, 80, 'Starwood\nLaunch\n2025', ha='center', fontsize=9, color=CORP_GREEN, fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_ylabel('Brazilian Rosewood Forest (% of 1960 levels)', fontsize=12, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Brazilian Rosewood: From Decline to Recovery', 
                 fontsize=16, fontweight='bold', color=CORP_NAVY, pad=20)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', facecolor=BG_WHITE, edgecolor=CORP_LIGHT_GRAY, fontsize=10)
    
    ax.text(0.5, 0.15, 'Neural tonewood technology can help reverse decades of deforestation',
            transform=ax.transAxes, fontsize=11, color=CORP_GREEN, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#C6F6D5', edgecolor=CORP_GREEN, alpha=0.8))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(CORP_LIGHT_GRAY)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_corporate/11_deforestation_timeline.png', 
                dpi=150, facecolor=BG_LIGHT_GRAY, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 11_deforestation_timeline.png")

# =============================================================================
# Run all chart generators
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("STARWOOD GUITARFX VISUALIZATIONS")
    print("CORPORATE PROFESSIONAL THEME")
    print("=" * 70)
    
    create_market_growth_chart()
    create_market_segments_pie()
    create_segment_growth_comparison()
    create_tam_sam_som()
    create_revenue_projection()
    create_competitive_landscape()
    create_conservation_impact()
    create_environmental_comparison()
    create_cites_status()
    create_ethical_summary()
    create_deforestation_timeline()
    
    print("=" * 70)
    print("All 11 corporate-themed visualizations generated!")
    print("Output: /home/ubuntu/Starwood/visualizations/output_corporate/")
    print("=" * 70)
