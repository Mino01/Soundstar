#!/usr/bin/env python3
"""
Starwood GuitarFX Market Opportunity Visualizations
Creates professional charts showing market size, growth, and segment analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

# Apply style first, then set custom fonts
plt.style.use('dark_background')

# Set custom styling after applying the style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.facecolor': '#0D0D0D',
    'figure.facecolor': '#0D0D0D',
    'text.color': '#F5DEB3',
    'axes.labelcolor': '#F5DEB3',
    'xtick.color': '#E8E8E8',
    'ytick.color': '#E8E8E8',
    'axes.edgecolor': '#8B4513',
    'grid.color': '#333333',
    'grid.alpha': 0.3,
})

# Color palette matching presentation
GOLD = '#DAA520'
WHEAT = '#F5DEB3'
ROSEWOOD = '#8B4513'
DARK_BG = '#0D0D0D'
LIGHT_TEXT = '#E8E8E8'
GREEN = '#4CAF50'

# Create output directory
os.makedirs('/home/ubuntu/Starwood/visualizations/output', exist_ok=True)

# =============================================================================
# Chart 1: Market Size & Growth Projection (2024-2030)
# =============================================================================
def create_market_growth_chart():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
    # 6% annual growth from $1.2B base
    market_size = [1.2, 1.27, 1.35, 1.43, 1.52, 1.61, 1.70]
    # AI/Neural segment growing at 12% CAGR
    ai_segment = [0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.24]
    
    # Create gradient effect with multiple bars
    bar_width = 0.35
    x = np.arange(len(years))
    
    # Total market bars
    bars1 = ax.bar(x - bar_width/2, market_size, bar_width, 
                   color=ROSEWOOD, edgecolor=GOLD, linewidth=1.5,
                   label='Total Guitar Effects Market')
    
    # AI/Neural segment bars
    bars2 = ax.bar(x + bar_width/2, ai_segment, bar_width,
                   color=GOLD, edgecolor=WHEAT, linewidth=1.5,
                   label='AI/Neural Effects Segment')
    
    # Add value labels on bars
    for bar, val in zip(bars1, market_size):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'${val:.2f}B', ha='center', va='bottom', fontsize=10,
                color=WHEAT, fontweight='bold')
    
    for bar, val in zip(bars2, ai_segment):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'${val*1000:.0f}M', ha='center', va='bottom', fontsize=9,
                color=GOLD, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_ylabel('Market Size (Billions USD)', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_title('Guitar Effects Market Growth Projection (2024-2030)', 
                 fontsize=18, fontweight='bold', color=WHEAT, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor=GOLD)
    
    # Add growth rate annotations
    ax.annotate('6% CAGR', xy=(5.5, 1.55), fontsize=11, color=ROSEWOOD, fontweight='bold')
    ax.annotate('12% CAGR', xy=(5.5, 0.22), fontsize=11, color=GOLD, fontweight='bold')
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output/01_market_growth_projection.png', 
                dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 01_market_growth_projection.png")

# =============================================================================
# Chart 2: Market Segments Breakdown (Pie Chart)
# =============================================================================
def create_market_segments_pie():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    segments = ['Distortion/Overdrive', 'Multi-FX Units', 'Delay/Reverb', 
                'Modulation', 'AI/Neural Effects', 'Acoustic Effects', 'Other']
    sizes = [320, 280, 240, 180, 120, 60, 50]  # in millions
    
    # Colors - highlight AI/Neural segment
    colors = [ROSEWOOD, '#6B3A2E', '#A0522D', '#8B6914', GOLD, '#D4AF37', '#4a4a4a']
    explode = (0, 0, 0, 0, 0.1, 0.05, 0)  # Explode AI/Neural segment
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=segments, 
                                       colors=colors, autopct='%1.1f%%',
                                       startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(edgecolor=DARK_BG, linewidth=2))
    
    # Style the text
    for text in texts:
        text.set_color(WHEAT)
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color(DARK_BG)
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.45, fc=DARK_BG, ec=GOLD, linewidth=2)
    ax.add_patch(centre_circle)
    
    # Add center text
    ax.text(0, 0.05, '$1.2B', ha='center', va='center', fontsize=28, 
            fontweight='bold', color=GOLD)
    ax.text(0, -0.12, 'Total Market\n(2024)', ha='center', va='center', 
            fontsize=12, color=LIGHT_TEXT)
    
    ax.set_title('Guitar Effects Market Segments (2024)', 
                 fontsize=18, fontweight='bold', color=WHEAT, pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output/02_market_segments_pie.png', 
                dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 02_market_segments_pie.png")

# =============================================================================
# Chart 3: Segment Growth Rates Comparison
# =============================================================================
def create_segment_growth_comparison():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    segments = ['AI/Neural\nEffects', 'Acoustic\nEffects', 'Multi-FX\nUnits', 
                'Delay/\nReverb', 'Modulation', 'Distortion/\nOverdrive']
    growth_rates = [12.0, 8.5, 7.2, 5.8, 4.5, 3.2]
    
    # Color bars based on growth rate
    colors = [GOLD if rate >= 8 else ROSEWOOD for rate in growth_rates]
    
    bars = ax.barh(segments, growth_rates, color=colors, edgecolor=WHEAT, linewidth=1)
    
    # Add value labels
    for bar, rate in zip(bars, growth_rates):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{rate}%', ha='left', va='center', fontsize=12,
                color=GOLD if rate >= 8 else WHEAT, fontweight='bold')
    
    # Add average line
    avg_growth = 6.0
    ax.axvline(x=avg_growth, color=GREEN, linestyle='--', linewidth=2, alpha=0.8)
    ax.text(avg_growth + 0.2, len(segments) - 0.5, f'Market Avg: {avg_growth}%', 
            color=GREEN, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Annual Growth Rate (%)', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_title('Growth Rate by Market Segment (CAGR 2024-2030)', 
                 fontsize=18, fontweight='bold', color=WHEAT, pad=20)
    ax.set_xlim(0, 15)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output/03_segment_growth_rates.png', 
                dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 03_segment_growth_rates.png")

# =============================================================================
# Chart 4: Target Addressable Market (TAM/SAM/SOM)
# =============================================================================
def create_tam_sam_som():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Concentric circles for TAM, SAM, SOM
    tam = 1200  # $1.2B - Total guitar effects market
    sam = 180   # $180M - AI/Neural + Acoustic effects
    som = 36    # $36M - Realistic 5-year capture (20% of SAM)
    
    # Create concentric circles
    circle_tam = plt.Circle((0.5, 0.5), 0.45, color=ROSEWOOD, alpha=0.4)
    circle_sam = plt.Circle((0.5, 0.5), 0.28, color=GOLD, alpha=0.5)
    circle_som = plt.Circle((0.5, 0.5), 0.12, color=WHEAT, alpha=0.7)
    
    ax.add_patch(circle_tam)
    ax.add_patch(circle_sam)
    ax.add_patch(circle_som)
    
    # Add labels
    ax.text(0.5, 0.92, 'TAM: $1.2B', ha='center', va='center', fontsize=14, 
            color=WHEAT, fontweight='bold')
    ax.text(0.5, 0.88, 'Total Guitar Effects Market', ha='center', va='center', 
            fontsize=10, color=LIGHT_TEXT)
    
    ax.text(0.5, 0.72, 'SAM: $180M', ha='center', va='center', fontsize=14, 
            color=GOLD, fontweight='bold')
    ax.text(0.5, 0.68, 'AI/Neural + Acoustic Segment', ha='center', va='center', 
            fontsize=10, color=LIGHT_TEXT)
    
    ax.text(0.5, 0.5, 'SOM\n$36M', ha='center', va='center', fontsize=16, 
            color=DARK_BG, fontweight='bold')
    ax.text(0.5, 0.42, '5-Year Target', ha='center', va='center', 
            fontsize=9, color=DARK_BG)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('Starwood GuitarFX Addressable Market', 
                 fontsize=18, fontweight='bold', color=WHEAT, pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=ROSEWOOD, alpha=0.4, label='TAM - Total Addressable Market'),
        mpatches.Patch(facecolor=GOLD, alpha=0.5, label='SAM - Serviceable Addressable Market'),
        mpatches.Patch(facecolor=WHEAT, alpha=0.7, label='SOM - Serviceable Obtainable Market')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              facecolor='#1a1a1a', edgecolor=GOLD, ncol=1)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output/04_tam_sam_som.png', 
                dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 04_tam_sam_som.png")

# =============================================================================
# Chart 5: Revenue Projection (5-Year)
# =============================================================================
def create_revenue_projection():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    
    # Revenue streams (in millions)
    plugin_revenue = [0.5, 1.8, 3.5, 5.2, 7.0]
    mobile_revenue = [0.2, 0.8, 1.5, 2.5, 3.5]
    hardware_revenue = [0, 0.5, 3.0, 8.0, 15.0]
    
    x = np.arange(len(years))
    width = 0.6
    
    # Stacked bar chart
    bars1 = ax.bar(x, plugin_revenue, width, label='FX Plugin ($149)', color=ROSEWOOD, edgecolor=GOLD)
    bars2 = ax.bar(x, mobile_revenue, width, bottom=plugin_revenue, label='FX Mobile ($49)', color='#A0522D', edgecolor=GOLD)
    bars3 = ax.bar(x, hardware_revenue, width, bottom=np.array(plugin_revenue) + np.array(mobile_revenue), 
                   label='FX Hardware ($299)', color=GOLD, edgecolor=WHEAT)
    
    # Total revenue line
    total_revenue = [p + m + h for p, m, h in zip(plugin_revenue, mobile_revenue, hardware_revenue)]
    ax.plot(x, total_revenue, 'o-', color=WHEAT, linewidth=2, markersize=8, label='Total Revenue')
    
    # Add total labels
    for i, total in enumerate(total_revenue):
        ax.text(i, total + 0.5, f'${total:.1f}M', ha='center', va='bottom',
                fontsize=11, color=WHEAT, fontweight='bold')
    
    ax.set_xlabel('Timeline', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_ylabel('Revenue (Millions USD)', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_title('Starwood GuitarFX 5-Year Revenue Projection', 
                 fontsize=18, fontweight='bold', color=WHEAT, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor=GOLD)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output/05_revenue_projection.png', 
                dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 05_revenue_projection.png")

# =============================================================================
# Chart 6: Competitive Landscape
# =============================================================================
def create_competitive_landscape():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Competitors positioned by Price (x) vs Feature Richness (y)
    competitors = {
        'TonewoodAmp2': (249, 40, 300),
        'Neural Amp Modeler': (0, 70, 200),
        'Quad Cortex': (1599, 90, 400),
        'Line 6 Helix': (1499, 85, 350),
        'Boss GT-1000': (999, 75, 300),
        'Starwood GuitarFX': (199, 95, 500),  # Our product - highlighted
    }
    
    for name, (price, features, size) in competitors.items():
        if name == 'Starwood GuitarFX':
            ax.scatter(price, features, s=size, c=GOLD, edgecolors=WHEAT, 
                      linewidths=3, alpha=0.9, zorder=5)
            ax.annotate(name, (price, features), xytext=(price + 80, features + 3),
                       fontsize=12, color=GOLD, fontweight='bold')
        else:
            ax.scatter(price, features, s=size, c=ROSEWOOD, edgecolors=LIGHT_TEXT, 
                      linewidths=1.5, alpha=0.7)
            ax.annotate(name, (price, features), xytext=(price + 50, features - 5),
                       fontsize=10, color=LIGHT_TEXT)
    
    # Add quadrant labels
    ax.axhline(y=60, color=GOLD, linestyle='--', alpha=0.3)
    ax.axvline(x=500, color=GOLD, linestyle='--', alpha=0.3)
    
    ax.text(100, 98, 'HIGH VALUE\n(Low Price, High Features)', fontsize=10, 
            color=GREEN, ha='center', fontweight='bold', alpha=0.8)
    ax.text(1200, 98, 'PREMIUM\n(High Price, High Features)', fontsize=10, 
            color=LIGHT_TEXT, ha='center', alpha=0.6)
    ax.text(100, 25, 'BUDGET\n(Low Price, Low Features)', fontsize=10, 
            color=LIGHT_TEXT, ha='center', alpha=0.6)
    ax.text(1200, 25, 'OVERPRICED\n(High Price, Low Features)', fontsize=10, 
            color='#ff6b6b', ha='center', alpha=0.6)
    
    ax.set_xlabel('Price (USD)', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_ylabel('Feature Richness Score', fontsize=12, fontweight='bold', color=WHEAT)
    ax.set_title('Competitive Landscape: Price vs. Features', 
                 fontsize=18, fontweight='bold', color=WHEAT, pad=20)
    ax.set_xlim(-100, 1800)
    ax.set_ylim(0, 105)
    
    # Add note about Starwood's unique position
    ax.text(800, 10, '★ Starwood offers the highest feature score at an accessible price point',
            fontsize=11, color=GOLD, style='italic')
    
    ax.grid(True, linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Starwood/visualizations/output/06_competitive_landscape.png', 
                dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created: 06_competitive_landscape.png")

# =============================================================================
# Run all chart generators
# =============================================================================
if __name__ == '__main__':
    print("Generating Starwood GuitarFX Market Opportunity Visualizations...")
    print("=" * 60)
    
    create_market_growth_chart()
    create_market_segments_pie()
    create_segment_growth_comparison()
    create_tam_sam_som()
    create_revenue_projection()
    create_competitive_landscape()
    
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("Output directory: /home/ubuntu/Starwood/visualizations/output/")
