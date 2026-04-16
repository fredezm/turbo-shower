import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# --- Setup ---
csv_name = '4rw_512perc_10kl_260k'
file_name = f'{csv_name}.csv'

# Load data
df = pd.read_csv(file_name, sep=',', decimal='.', index_col=False)

dim_names = [
    'Temperatura ambiente', 
    'P-iqb', 
    'P-eletrico', 
    'P-agua', 
    'P-gas', 
    'IQB total', 
    'Custo elétrico total', 
    'Custo de gás total', 
    'Custo de água total'
]

all_cols_to_use = dim_names
color_col = 'Temperatura ambiente'

# 1. Clean Data
for col in all_cols_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=all_cols_to_use)

if df_clean.empty:
    print("Warning: All data rows were dropped.")
else:
    # --- Matplotlib Plotting ---
    
    # 2. Prepare Data for Parallel Coordinates
    # We must normalize all columns to 0-1 range so they fit on one graph,
    # but we will label the top/bottom with real values later.
    df_norm = df_clean[dim_names].copy()
    
    # Store min/max for labels later
    min_max_vals = {}
    
    for col in dim_names:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        min_max_vals[col] = (min_val, max_val)
        
        # Normalize: (x - min) / (max - min)
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5 # Default middle if all values are same

    # 3. Create Line Segments efficiently
    # We create a 3D array: (Number of Lines, Number of Axes, 2 (x,y))
    num_lines = len(df_norm)
    num_dims = len(dim_names)
    
    # X coordinates (0, 1, 2, 3...)
    x = np.arange(num_dims)
    
    # Create the 3D array for the segments
    segments = np.zeros((num_lines, num_dims, 2))
    segments[:, :, 0] = x  # Set X coordinates repeated for every row
    segments[:, :, 1] = df_norm.values # Set Y coordinates (normalized values)

    # 4. Create the Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # LineCollection is much faster than plotting individual lines
    lc = LineCollection(segments, cmap='turbo', alpha=0.3)
    lc.set_array(df_clean[color_col].values) # Color mapping based on actual Temp
    lc.set_linewidth(1)
    
    ax.add_collection(lc)

    # 5. Customize Axes to look like Parallel Coordinates
    ax.set_xlim(-0.5, num_dims - 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    # Remove standard Y ticks (since every vertical axis has different scales)
    ax.set_yticks([])
    
    # Set X ticks as column names
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=30, ha='right')
    
    # Add vertical lines and "Min/Max" labels for every dimension
    for i, col in enumerate(dim_names):
        mn, mx = min_max_vals[col]
        
        # Draw vertical axis line
        ax.plot([i, i], [0, 1], color='black', alpha=0.2, linewidth=1, zorder=0)
        
        # Add text labels for the scale (formatted to 2 decimals)
        ax.text(i, -0.02, f"{mn:.2f}", ha='center', va='top', fontsize=9, fontweight='bold')
        ax.text(i, 1.02, f"{mx:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add Colorbar
    cbar = plt.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label('Temperatura Ambiente')

    ax.set_title('Effect of Weights and Temperature on Total Costs (Parallel Coordinates)')
    
    # Save
    output_file = f'{csv_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as {output_file} with {len(df_clean)} rows.")