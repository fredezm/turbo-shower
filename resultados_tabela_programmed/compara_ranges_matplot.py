import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_graph(target, name_file, file1_name, file2_name):

    df1 = pd.read_csv(f"{file1_name}.csv")
    df2 = pd.read_csv(f"{file2_name}.csv")

    # Column you want to analyse
    column = "Temperatura ambiente"
    # All numerical columns except the grouping column
    value_columns = [
    "P-iqb", "P-eletrico", "P-agua", "P-gas",
    "Temperatura ambiente", "Tarifa da energia Selétrica",
    "IQB 1", "IQB 2", "IQB 3", "IQB 4", "IQB 5", "IQB 6", "IQB 7",
    "IQB médio", "IQB total", #"Recompensa total",
    "Custo elétrico total", "Custo de gás total",
    "Custo de água total", "Custo total do banho"
    ]

    # Remove the grouping column from the summary list
    value_columns = [c for c in value_columns if c != column]

    # Compute min and max grouped by temperature
    min1 = df1.groupby(column)[value_columns].min()
    max1 = df1.groupby(column)[value_columns].max()
    min2 = df2.groupby(column)[value_columns].min()
    max2 = df2.groupby(column)[value_columns].max()

    # --- Matplotlib Plotting ---
    
    # Setup the x-axis labels and positions
    labels = min1.index.astype(str)
    x = np.arange(len(labels))
    width = 0.20  # The width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting the 4 bars side-by-side
    # We offset the 'x' position for each series
    rects1 = ax.bar(x - 1.5*width, min1[target], width, label=f"{file1_name} - Min", color="#00BECC")
    rects2 = ax.bar(x - 0.5*width, min2[target], width, label=f"{file2_name} - Min", color="#74CC00")
    rects3 = ax.bar(x + 0.5*width, max1[target], width, label=f"{file1_name} - Max", color="#FCE564")
    rects4 = ax.bar(x + 1.5*width, max2[target], width, label=f"{file2_name} - Max", color="#F03919")

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(target)
    ax.set_xlabel('Weather Temperature')
    ax.set_title(f"Min/Max of '{target}' by Weather Temperature")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Optional: Add a grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save as PNG
    output_filename = f"{name_file}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    
    print(f"Saved as {output_filename}")

file1_name = "4rw_512perc_10kl_260k"
file2_name = "4rw_10kl_260k"

create_graph("IQB total", "compara_ranges/comp_iqb", file1_name, file2_name)
create_graph("Custo elétrico total", "compara_ranges/comp_ele", file1_name, file2_name)
create_graph("Custo de gás total", "compara_ranges/comp_gas", file1_name, file2_name)
create_graph("Custo de água total", "compara_ranges/comp_agua", file1_name, file2_name)
create_graph("Custo total do banho", "compara_ranges/comp_custo", file1_name, file2_name)