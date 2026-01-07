#!/usr/bin/env python3
"""
Starwood GuitarFX Ethical & Sustainability Visualizations
Highlighting the environmental benefits of neural tonewood technology.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
import os

# Create output directory
os.makedirs('/home/ubuntu/Starwood/visualizations/output_tonewood', exist_ok=True)

# =============================================================================
# TONEWOOD & SUSTAINABILITY COLOR PALETTE
# =============================================================================
BG_CREAM = '#FDF5E6'
BG_WARM = '#FAF0E6'

# Tonewoods
BRAZILIAN_ROSEWOOD = '#4A2C2A'
COCOBOLO = '#8B4513'
HONDURAN_MAHOGANY = '#6B3A2E'
KOA = '#B8860B'
EBONY = '#2C2416'
MAPLE = '#F5DEB3'

# Sustainability colors
FOREST_GREEN = '#228B22'
LEAF_GREEN = '#32CD32'
EARTH_BROWN = '#8B4513'
SKY_BLUE = '#87CEEB'
DANGER_RED = '#DC143C'

# Text
TEXT_DARK = '#2C2416'
TEXT_MEDIUM = '#5D4037'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
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
# Chart 7: Endangered Tonewood Conservation Impact
# =============================================================================
def create_conservation_impact():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    # Data: Trees saved per year if X% of guitarists use Starwood
    adoption_rates = ['5%', '10%', '25%', '50%', '75%']
    trees_saved = [2500, 5000, 12500, 25000, 37500]  # Estimated trees per year
    
    # Create bars with gradient effect
    colors = [LEAF_GREEN, FOREST_GREEN, '#2E8B57', '#006400', '#004d00']
    
    bars = ax.bar(adoption_rates, trees_saved, color=colors, edgecolor=EBONY, linewidth=2, width=0.6)
    
    # Add tree icons as value labels
    for bar, trees in zip(bars, trees_saved):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
                f'{trees:,}\nTrees', ha='center', va='bottom', fontsize=11,
                color=FOREST_GREEN, fontweight='bold')
    
    ax.set_xlabel('Starwood Adoption Rate Among Guitarists', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_ylabel('Tonewood Trees Saved Per Year', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('CONSERVATION IMPACT: Trees Saved Through Neural Tonewood Technology', 
                 fontsize=16, fontweight='bold', color=FOREST_GREEN, pad=20)
    ax.set_ylim(0, 45000)
    
    # Add annotation about Brazilian Rosewood
    ax.text(0.5, 0.92, 'Brazilian Rosewood: CITES Appendix I Protected (Critically Endangered)',
            transform=ax.transAxes, fontsize=11, color=DANGER_RED, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE4E1', edgecolor=DANGER_RED, alpha=0.8))
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(FOREST_GREEN)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/07_conservation_impact.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 07_conservation_impact.png")

# =============================================================================
# Chart 8: Traditional vs Starwood Environmental Comparison
# =============================================================================
def create_environmental_comparison():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    categories = ['Trees\nHarvested', 'CO2\nFootprint\n(kg)', 'Water\nUsage\n(liters)', 
                  'Habitat\nDestruction\n(sq m)', 'Years to\nRegrow']
    
    # Traditional premium guitar (Brazilian Rosewood)
    traditional = [3, 150, 2000, 50, 100]  # Normalized/estimated values
    
    # Starwood GuitarFX
    starwood = [0, 5, 10, 0, 0]  # Minimal digital footprint
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, traditional, width, label='Traditional Premium Guitar', 
                   color=DANGER_RED, edgecolor=EBONY, linewidth=2, alpha=0.8)
    bars2 = ax.bar(x + width/2, starwood, width, label='Starwood GuitarFX', 
                   color=FOREST_GREEN, edgecolor=EBONY, linewidth=2, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, traditional):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val}', ha='center', va='bottom', fontsize=10,
                    color=DANGER_RED, fontweight='bold')
    
    for bar, val in zip(bars2, starwood):
        label = 'ZERO' if val == 0 else str(val)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha='center', va='bottom', fontsize=10,
                color=FOREST_GREEN, fontweight='bold')
    
    ax.set_ylabel('Environmental Impact (Various Units)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('ENVIRONMENTAL FOOTPRINT: Traditional Guitar vs. Starwood', 
                 fontsize=16, fontweight='bold', color=TEXT_DARK, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', facecolor=BG_CREAM, edgecolor=COCOBOLO, fontsize=11)
    ax.set_ylim(0, 180)
    
    # Add reduction percentage
    ax.text(0.5, 0.88, '97% REDUCTION in Environmental Impact',
            transform=ax.transAxes, fontsize=14, color=FOREST_GREEN, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=FOREST_GREEN, alpha=0.9))
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/08_environmental_comparison.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 08_environmental_comparison.png")

# =============================================================================
# Chart 9: CITES Protected Tonewoods Status
# =============================================================================
def create_cites_status():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    tonewoods = ['Brazilian\nRosewood', 'Cocobolo', 'Honduran\nMahogany', 
                 'African\nBlackwood', 'Ebony', 'Bubinga']
    
    # CITES status (1=Appendix I Critical, 2=Appendix II Threatened, 3=Appendix III Monitored)
    cites_level = [1, 2, 2, 2, 2, 2]
    
    # Price premium for traditional guitars using these woods ($)
    price_premium = [15000, 3000, 1500, 2500, 2000, 1800]
    
    # Colors based on endangerment
    colors = [DANGER_RED if level == 1 else '#FF8C00' if level == 2 else KOA for level in cites_level]
    
    bars = ax.barh(tonewoods, price_premium, color=colors, edgecolor=EBONY, linewidth=2, height=0.6)
    
    # Add CITES labels
    cites_labels = ['CITES I\nCRITICAL', 'CITES II\nTHREATENED', 'CITES II\nTHREATENED',
                    'CITES II\nTHREATENED', 'CITES II\nTHREATENED', 'CITES II\nTHREATENED']
    
    for bar, label, premium in zip(bars, cites_labels, price_premium):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f'${premium:,}', ha='left', va='center', fontsize=11,
                color=TEXT_DARK, fontweight='bold')
        ax.text(500, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=9,
                color='white', fontweight='bold')
    
    ax.set_xlabel('Price Premium for Traditional Guitar ($)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('CITES PROTECTED TONEWOODS: Endangered Species & Price Premiums', 
                 fontsize=16, fontweight='bold', color=DANGER_RED, pad=20)
    ax.set_xlim(0, 18000)
    
    # Add Starwood solution box
    ax.text(0.65, 0.15, 'STARWOOD SOLUTION:\nAll premium tones\nZERO endangered wood\n$149-$299',
            transform=ax.transAxes, fontsize=12, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=FOREST_GREEN, edgecolor=EBONY, alpha=0.95))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=DANGER_RED, edgecolor=EBONY, label='CITES Appendix I - Critically Endangered'),
        mpatches.Patch(facecolor='#FF8C00', edgecolor=EBONY, label='CITES Appendix II - Threatened'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', facecolor=BG_CREAM, edgecolor=COCOBOLO)
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/09_cites_status.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 09_cites_status.png")

# =============================================================================
# Chart 10: Ethical Value Proposition Summary
# =============================================================================
def create_ethical_summary():
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    # Create a visual summary with key metrics
    metrics = [
        ('ZERO', 'Endangered Trees Cut', FOREST_GREEN),
        ('100%', 'Digital & Sustainable', LEAF_GREEN),
        ('$14,850', 'Savings vs. Brazilian\nRosewood Guitar', KOA),
        ('97%', 'Lower Carbon\nFootprint', SKY_BLUE),
        ('INFINITE', 'Tonewood Variety\nWithout Harm', BRAZILIAN_ROSEWOOD),
    ]
    
    # Create circular badges for each metric
    positions = [(0.15, 0.7), (0.5, 0.7), (0.85, 0.7), (0.3, 0.3), (0.7, 0.3)]
    
    for (value, label, color), (x, y) in zip(metrics, positions):
        # Draw circle
        circle = Circle((x, y), 0.12, fc=color, ec=EBONY, linewidth=3, transform=ax.transAxes)
        ax.add_patch(circle)
        
        # Add value text
        ax.text(x, y + 0.02, value, transform=ax.transAxes, ha='center', va='center',
                fontsize=16 if len(value) < 5 else 12, fontweight='bold', color='white')
        
        # Add label below
        ax.text(x, y - 0.18, label, transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold', color=TEXT_DARK)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'STARWOOD ETHICAL VALUE PROPOSITION', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=20, fontweight='bold', color=BRAZILIAN_ROSEWOOD)
    
    ax.text(0.5, 0.88, 'Premium Guitar Tone Without Environmental Cost', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, color=TEXT_MEDIUM, style='italic')
    
    # Bottom message
    ax.text(0.5, 0.08, '"Every Starwood user helps preserve endangered forests\nwhile enjoying the world\'s finest guitar tones."',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color=FOREST_GREEN, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=FOREST_GREEN, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/10_ethical_summary.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 10_ethical_summary.png")

# =============================================================================
# Chart 11: Deforestation Timeline & Starwood Impact
# =============================================================================
def create_deforestation_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_WARM)
    ax.set_facecolor(BG_CREAM)
    
    years = [1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
    
    # Brazilian Rosewood availability (% of original forest)
    rosewood_decline = [100, 85, 60, 35, 15, 8, 5, 3]
    
    # Projected with Starwood adoption (starting 2025)
    rosewood_with_starwood = [100, 85, 60, 35, 15, 8, 5, 7]  # Recovery begins
    
    # Plot decline
    ax.fill_between(years[:7], rosewood_decline[:7], alpha=0.3, color=DANGER_RED)
    ax.plot(years[:7], rosewood_decline[:7], 'o-', color=DANGER_RED, linewidth=3, 
            markersize=8, label='Historical Decline')
    
    # Plot projected recovery with Starwood
    ax.fill_between(years[5:], rosewood_with_starwood[5:], alpha=0.3, color=FOREST_GREEN)
    ax.plot(years[5:], rosewood_with_starwood[5:], 'o--', color=FOREST_GREEN, linewidth=3, 
            markersize=8, label='Projected with Starwood Adoption')
    
    # Mark CITES ban
    ax.axvline(x=1992, color=EBONY, linestyle=':', linewidth=2, alpha=0.7)
    ax.text(1992, 80, 'CITES\nBan\n1992', ha='center', fontsize=9, color=EBONY, fontweight='bold')
    
    # Mark Starwood launch
    ax.axvline(x=2025, color=FOREST_GREEN, linestyle=':', linewidth=2, alpha=0.7)
    ax.text(2025, 80, 'Starwood\nLaunch\n2025', ha='center', fontsize=9, color=FOREST_GREEN, fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_ylabel('Brazilian Rosewood Forest (% of 1960 levels)', fontsize=13, fontweight='bold', color=TEXT_DARK)
    ax.set_title('BRAZILIAN ROSEWOOD: From Decline to Recovery', 
                 fontsize=16, fontweight='bold', color=BRAZILIAN_ROSEWOOD, pad=20)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', facecolor=BG_CREAM, edgecolor=COCOBOLO, fontsize=11)
    
    # Add message
    ax.text(0.5, 0.15, 'Neural tonewood technology can help reverse decades of deforestation',
            transform=ax.transAxes, fontsize=12, color=FOREST_GREEN, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=FOREST_GREEN, alpha=0.8))
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color(COCOBOLO)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output_tonewood/11_deforestation_timeline.png', 
                dpi=150, facecolor=BG_WARM, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 11_deforestation_timeline.png")

# =============================================================================
# Run all ethical/sustainability chart generators
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("STARWOOD ETHICAL & SUSTAINABILITY VISUALIZATIONS")
    print("Preserving Forests Through Neural Tonewood Technology")
    print("=" * 70)
    
    create_conservation_impact()
    create_environmental_comparison()
    create_cites_status()
    create_ethical_summary()
    create_deforestation_timeline()
    
    print("=" * 70)
    print("All ethical/sustainability visualizations generated!")
    print("Output: /home/ubuntu/Starwood/visualizations/output_tonewood/")
    print("=" * 70)
