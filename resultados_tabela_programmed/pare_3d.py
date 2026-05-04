import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pareto_3d_by_temp():
    name_agent = '4rw_512perc_10kl_260k6'
    
    def identify_pareto(scores):
        n_points = scores.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        for i, c in enumerate(scores):
            if is_pareto[i]:
                # Assume-se que maior valor é melhor para todas as colunas. 
                # Se custos devem ser minimizados, multiplique-os por -1 antes de passar para esta função.
                is_pareto[is_pareto] = np.any(scores[is_pareto] > c, axis=1) | np.all(scores[is_pareto] == c, axis=1)
                is_pareto[i] = True
        return is_pareto
    
    df = pd.read_csv(f"resultados_gpi-ls_{name_agent}.csv")
    ax1, ax2, ax3 = 'Total IQB', 'Total Electric Cost', 'Total Water Cost'
    temp_col = 'Ambient Temperature'

    target_temps = [15,20,25,30]
    df = df[df['Ambient Temperature'].isin(target_temps)]
    
    # Verifica se o filtro retornou algo para evitar erros no plot
    if df.empty:
        print(f"Nenhum dado encontrado para as temperaturas: {target_temps}")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Cores para diferentes temperaturas
    temperatures = sorted(df[temp_col].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))

    for temp, color in zip(temperatures, colors):
        # Filtra dados para a temperatura específica
        temp_df = df[df[temp_col] == temp]
        recompensas = temp_df[[ax1, ax2, ax3]].values

        # Identifica a fronteira para este grupo
        mask = identify_pareto(recompensas)
        pareto_points = recompensas[mask]
        dominated_points = recompensas[~mask]

        # Plota pontos dominados (mais claros/transparentes)
        ax.scatter(dominated_points[:, 0], dominated_points[:, 1], dominated_points[:, 2], 
                   color=color, alpha=0.1, s=20)

        # Plota a Fronteira de Pareto para esta temperatura
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], 
                   color=color, label=f'Temp: {temp}', edgecolors='black', 
                   linewidths=0.5, s=60, alpha=0.7)

    ax.set_xlabel(ax1)
    ax.set_ylabel(ax2)
    ax.set_zlabel(ax3)
    ax.set_title(f'Pareto Frontiers by Temperature')
    
    # Legenda ajustada para não poluir o gráfico se houver muitas temperaturas
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    output_folder = "resultados_pf"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"pf_temp_{name_agent}.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_pareto_3d_by_temp()