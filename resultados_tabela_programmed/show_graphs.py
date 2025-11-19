import pandas as pd
import plotly.express as px

file_name = 'resultados_tabela.csv' 

# 1. Correct Loading:
# - sep=',': Explicitly set comma delimiter.
# - decimal='.': The file uses dots for decimals (e.g., 15.0).
# - index_col=False: Prevents pandas from mistaking the first columns as an index, fixing the shift.
df = pd.read_csv(file_name, sep=',', decimal='.', index_col=False)

# 2. Dimensions
dimensions = [
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

# 3. Clean Data
df_plot = df[dimensions].copy()

# Ensure numeric conversion (Temperatures should now be 15, 20, 25, 30)
for col in dimensions:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

# 4. Generate Plot
output_file = 'parallel_coordinates_plot_fixed_alignment.html'

# Remove rows where Temperature is NaN (only if any exist) to ensure clean axis
df_plot = df_plot.dropna(subset=['Temperatura ambiente'])

fig = px.parallel_coordinates(
    df_plot,
    dimensions=dimensions,
    color='Temperatura ambiente', 
    color_continuous_scale=px.colors.sequential.Turbo,
    title='Effect of Weights and Temperature on Total Costs (Corrected Data Alignment)'
)

fig.write_html(output_file)
print(f"Success! The corrected plot has been saved as '{output_file}'.")
print("The 'Temperatura ambiente' axis should now correctly show the range 15 to 30.")