"""
ICML Publication Style Template for Routesmith Paper

ICML Figure Standards:
- White background
- 12pt+ fonts (use matplotlib.rcParams)
- 300 DPI
- PNG format, 100KB-500KB

Fonts: 16pt for titles, 14pt for labels, 12pt for tick marks
Colors: Colorblind-friendly palette
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# ICML-compliant style settings
def apply_icml_style():
    """Apply ICML publication standards to matplotlib figures."""
    
    # Font settings (ICML requires 12pt+)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    
    # Title and label weights
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    # White background (ICML requirement)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    # Axis settings
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Grid options
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.8
    
    # DPI for high-quality output
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # LaTeX rendering (disabled for speed, use raw strings for math)
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    # Seaborn style
    sns.set_style("whitegrid")
    
    return plt


def get_colorblind_palette():
    """Return colorblind-friendly color palette."""
    # Okabe-Ito palette (colorblind-friendly)
    colors = [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#CC79A7',  # Reddish Purple
        '#000000',  # Black
    ]
    return colors


def setup_figure(figsize=(6, 4.5)):
    """Create a new figure with ICML styling."""
    apply_icml_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    return fig, ax


def add_grid(ax, alpha=0.3):
    """Add subtle grid to axis."""
    ax.grid(True, alpha=alpha, linestyle='--', linewidth=0.8)


def finalize_figure(fig, ax, title=None, xlabel=None, ylabel=None, 
                   legend=None, grid=True, save_path=None):
    """Finalize figure with common settings."""
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    if legend:
        ax.legend(**legend)
    
    if grid:
        add_grid(ax)
    
    fig.tight_layout()
    
    if save_path:
        # Ensure file size is reasonable (100KB-500KB)
        fig.savefig(save_path, dpi=300, format='png', 
                   facecolor='white', edgecolor='none')
        import os
        size_kb = os.path.getsize(save_path) / 1024
        print(f"Saved: {save_path} ({size_kb:.1f} KB)")
    
    return fig, ax


# Initialize style on import
apply_icml_style()

if __name__ == "__main__":
    print("ICML style loaded. Use apply_icml_style() or setup_figure().")
    print("Colorblind palette available via get_colorblind_palette()")
