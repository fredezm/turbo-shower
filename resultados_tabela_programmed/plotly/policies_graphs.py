import pandas as pd
import plotly.graph_objects as go

import pathlib

# --- Setup ---
csv_name = '../4rw_512perc_10kl_260k'
file_name = f'{csv_name}.csv'

# Load data
df = pd.read_csv(file_name, sep=',', decimal='.', index_col=False)

weights_split = df["Weights"].str.split(',', expand=True).astype(float)
weights_split.columns = ['W-iqb', 'W-electricity', 'W-water', 'W-gas']

# Drop the original packed column and join the new ones
df = df.drop(columns=["Weights"]).join(weights_split)

dim_names = [
    'Ambient Temperature', 
    'W-iqb', 
    'W-electricity', 
    'W-water', 
    'W-gas', 
    'Total IQB', 
    'Total Electric Cost', 
    'Total Gas Cost', 
    'Total Water Cost'
]

# dim_names = [
#     'Ambient Temperature', 
#     'W-iqb', 
#     'W-electricity', 
#     'W-water', 
#     'W-gas', 
#     "IQB 1",
#     "IQB 2",
#     "IQB 3",
#     "IQB 4",
#     "IQB 5",
#     "IQB 6",
#     "IQB 7",
#     'Total IQB',
# ]

# 💡 Recommended Fix: Identify all columns needed for the plot (dimensions + color)
all_cols_to_use = dim_names
color_col = 'Ambient Temperature'

# 1. Ensure all columns are numeric, coercing errors to NaN
# We do this for the entire DataFrame first.
for col in all_cols_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Key Step: Drop rows where ANY of the required columns is NaN.
# This ensures that Plotly has complete data for every line it draws.
df_clean = df.dropna(subset=all_cols_to_use)

# Check if any data remains
if df_clean.empty:
    print("Warning: All data rows were dropped due to missing or non-numeric values. Plot will be empty.")
    # You might want to skip plotting if the DataFrame is empty
else:
    # --- Plot Generation (using the cleaned df_clean) ---
    
    parcoords_dims = []
    for col in dim_names:
        series = df_clean[col] # Use the cleaned series
        min_val = series.min()
        max_val = series.max()
        
        parcoords_dims.append(dict(
            range = [min_val, max_val],
            label = col,
            values = series.tolist() # Convert to list for Plotly for safety
        ))

    # Use the cleaned color column values
    fig = go.Figure(data=go.Parcoords(
        line = dict(
            color = df_clean[color_col], 
            colorscale = 'Turbo',
            showscale = True,
            colorbar = dict(title='Temp')
        ),
        dimensions = parcoords_dims
    ))

    fig.update_layout(title='Effect of Weights and Temperature on Total Costs (Incl. Total Bath Cost)')
    fig.write_html(f'{csv_name}.html')

    print(f"Plot successfully generated with {len(df_clean)} rows of data.")