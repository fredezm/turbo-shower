import pandas as pd
import plotly.graph_objects as go



def create_graph(target, name_file, file1_name, file2_name):


    df1 = pd.read_csv(f"{file1_name}.csv")
    df2 = pd.read_csv(f"{file2_name}.csv")

    # Column you want to analyse
    column = "Ambient Temperature"
    # All numerical columns except the grouping column
    value_columns = [
        "Weights",
        "Ambient Temperature","Electricity Tariff",
        "IQB 1","IQB 2","IQB 3","IQB 4","IQB 5","IQB 6","IQB 7",
        "Average IQB","Total IQB",#"Total Reward",
        "Total Electric Cost","Total Gas Cost",
        "Total Water Cost","Total Bath Cost"
    ]

    # Remove the grouping column from the summary list
    value_columns = [c for c in value_columns if c != column]

    # Compute min and max grouped by temperature
    min1 = df1.groupby(column)[value_columns].min()
    max1 = df1.groupby(column)[value_columns].max()
    min2 = df2.groupby(column)[value_columns].min()
    max2 = df2.groupby(column)[value_columns].max()

    # Create Plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"{file1_name} - Min",
        x=min1.index.astype(str),
        y=min1[target],
        marker_color="#00BECC"
    ))

    fig.add_trace(go.Bar(
        name=f"{file2_name} - Min",
        x=min2.index.astype(str),
        y=min2[target],
        marker_color="#74CC00"
    ))

    fig.add_trace(go.Bar(
        name=f"{file1_name} - Max",
        x=max1.index.astype(str),
        y=max1[target],
        marker_color="#FCE564"
    ))

    fig.add_trace(go.Bar(
        name=f"{file2_name} - Max",
        x=max2.index.astype(str),
        y=max2[target],
        marker_color="#F03919"
    ))

    fig.update_layout(
        title=f"Min/Max of '{target}' by Ambient Temperature",
        xaxis_title="Ambient Temperature",
        yaxis_title=target,
        barmode="group"
    )

    fig.write_html(f"{name_file}.html")
    print(f"Saved as {name_file}.html")

file1_name = "4rw_512perc_10kl_60k"
file2_name = "4rw_512perc_10kl_260k"

create_graph("Total IQB", "compara_ranges/comp_iqb", file1_name, file2_name)
create_graph("Total Electric Cost", "compara_ranges/comp_ele", file1_name, file2_name)
create_graph("Total Gas Cost", "compara_ranges/comp_gas", file1_name, file2_name)
create_graph("Total Water Cost", "compara_ranges/comp_agua", file1_name, file2_name)
create_graph("Total Bath Cost", "compara_ranges/comp_custo", file1_name, file2_name)