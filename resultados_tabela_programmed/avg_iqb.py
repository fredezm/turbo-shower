import pandas as pd

for k_num in range(0, 10):
    df = pd.read_csv(f'resultados_gpi-ls_4rw_512perc_10kl_260k{k_num}.csv')
    
    # Overall mean across all temperatures
    grouped_df = df.groupby(['Weights'])['Average IQB'].mean().reset_index()
    grouped_df.rename(columns={'Average IQB': 'Mean_Average_IQB'}, inplace=True)
    
    # Average IQB per temperature (15, 20, 25, 30)
    for temp in [15, 20, 25, 30]:
        temp_mean = (
            df[df['Ambient Temperature'] == temp]
            .groupby(['Weights'])['Average IQB']
            .mean()
            .reset_index()
            .rename(columns={'Average IQB': f'Mean_IQB_temp{temp}'})
        )
        grouped_df = grouped_df.merge(temp_mean, on=['Weights'], how='left')
    
    grouped_df.to_csv(f'avg_260k{k_num}.csv', index=False)
    print(grouped_df)