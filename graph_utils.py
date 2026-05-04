import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def obter_pontos_pareto(df):
    """
    Retorna apenas os pontos não dominados (Fronteira de Pareto).
    Objetivo: Minimizar 'Total Water Cost' e Maximizar 'Total IQB'.
    """
    # 1. Ordenar por custo (ascendente) e depois por qualidade (descendente)
    df_sorted = df.sort_values(by=["Total Water Cost", "Total IQB"], 
                                ascending=[True, False])

    pareto_front = []
    max_qualidade_ate_agora = -1.0

    for _, row in df_sorted.iterrows():
        # Se este ponto traz uma qualidade melhor do que todos os pontos mais baratos que ele,
        # ele é um ponto de Pareto.
        if row["Total IQB"] > max_qualidade_ate_agora:
            pareto_front.append(row)
            max_qualidade_ate_agora = row["Total IQB"]
        
    return pd.DataFrame(pareto_front)

def plot_fronteira_pareto(df_resultados, folder_path):
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)
    temperaturas = df_resultados["Ambient Temperature"].unique()

    for temp in temperaturas:
        df_temp = df_resultados[df_resultados["Ambient Temperature"] == temp]
        
        # --- FILTRAGEM ---
        df_pareto = obter_pontos_pareto(df_temp)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_pareto, x="Total Water Cost", y="Total IQB", 
                        s=120, color="steelblue", edgecolor="#002147", linewidth=1.0, alpha=0.9, zorder=3)

        # Anota apenas os pesos dos pontos ótimos
        # for _, row in df_pareto.iterrows():
        #     plt.annotate(row["Pesos"], (row["Total Water Cost"], row["Total IQB"]),
        #                  xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.title(f"Pareto Frontier - Ambient Temp: {temp}°C")
        plt.xlabel("Total Water Cost (R$)")
        plt.ylabel("Total Bathing Quality Index (IQB)")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.gca().invert_xaxis()
        
        temp_str = str(temp).replace(".", "-")
        plt.savefig(os.path.join(path_pareto, f"pareto_T{temp_str}.png"), dpi=150)
        plt.close()

def plot_pareto_multiplas_temperaturas(df_resultados, folder_path, temps_escolhidas):
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)
    temperaturas = df_resultados["Ambient Temperature"].unique()

    plt.figure(figsize=(10, 6))

    cores = ["steelblue", "darkorange", "crimson"]  

    for i, temp in enumerate(temps_escolhidas):
        df_temp = df_resultados[df_resultados["Ambient Temperature"] == temp]
        
        df_pareto = obter_pontos_pareto(df_temp)

        sns.scatterplot(
            data=df_pareto,
            x="Total Water Cost",
            y="Total IQB",
            s=120,
            color=cores[i],
            label=f"{temp}°C",
            edgecolor="#002147",
            linewidth=1.0,
            alpha=0.9,
            zorder=3
        )

    plt.title("Pareto Frontiers for Selected Temperatures", fontsize=16)
    plt.xlabel("Total Water Cost (R$)", fontsize=15)
    plt.ylabel("Average Bathing Quality Index (IQB)", fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Ambient Temp", fontsize=12, title_fontsize=13)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=13)

    plt.gca().invert_xaxis()

    plt.savefig(os.path.join(path_pareto, "pareto_multiplas_temperaturas.png"), dpi=150)
    plt.close()

def plot_comparativo_pareto_clima(df_resultados, folder_path, temp_fria, temp_amena, temp_quente):
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)

    climas = {
        temp_fria: {"label": f"Fria ({temp_fria}°C)", "color": "blue"},
        temp_amena: {"label": f"Amena ({temp_amena}°C)", "color": "orange"},
        temp_quente: {"label": f"Quente ({temp_quente}°C)", "color": "red"}
    }

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    for temp, config in climas.items():
        df_temp = df_resultados[df_resultados["Ambient Temperature"] == temp]
        if df_temp.empty: continue

        # --- FILTRAGEM ---
        df_pareto = obter_pontos_pareto(df_temp)

        plt.scatter(df_pareto["Total Water Cost"], df_pareto["Total IQB"], 
                    s=120, color=config["color"], label=config["label"], edgecolor="black", zorder=3)

        plt.plot(df_pareto["Total Water Cost"], df_pareto["Total IQB"], 
                 color=config["color"], linestyle="-", linewidth=2, alpha=0.6, zorder=2)

    plt.title("Comparação de Fronteiras de Pareto por Clima (Apenas Pontos Ótimos)", fontsize=14, fontweight='bold')
    plt.xlabel("Total Water Cost (R$)", fontsize=12)
    plt.ylabel("Total IQB (Qualidade)", fontsize=12)
    plt.legend(title="Ambient Temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(path_pareto, "comparativo_pareto_climas.png"), dpi=200)

def plot_fronteira_pareto_global(df_resultados, folder_path):
    """
    Gera um gráfico único com as fronteiras de Pareto por temperatura.
    Cores: RdYlBu_r
    """
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)

    # 1. Filtrar Pareto para cada temperatura individualmente
    df_pareto_global = pd.DataFrame()
    for temp in df_resultados["Ambient Temperature"].unique():
        df_temp = df_resultados[df_resultados["Ambient Temperature"] == temp]
        df_p = obter_pontos_pareto(df_temp) # Função de filtragem já definida anteriormente
        df_pareto_global = pd.concat([df_pareto_global, df_p])

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # 2. Definir a paleta cromática (RdYlBu_r: Blue-Yellow-Red)
    cmap_name = "RdYlBu_r"

    # 3. Criar o scatter plot com o novo gradiente
    scatter = sns.scatterplot(
        data=df_pareto_global, 
        x="Total Water Cost", 
        y="Total IQB", 
        hue="Ambient Temperature", 
        palette=cmap_name, 
        s=100, 
        edgecolor="black", 
        alpha=0.9, 
        zorder=3
    )

    # 4. Desenhar as linhas de conexão com a mesma cor dos pontos
    # Criamos um mapeador de cores para garantir que a linha tenha a cor exata do ponto
    cores = sns.color_palette(cmap_name, as_cmap=True)
    norm = plt.Normalize(
        df_pareto_global["Ambient Temperature"].min(), 
        df_pareto_global["Ambient Temperature"].max()
    )

    for temp in sorted(df_pareto_global["Ambient Temperature"].unique()):
        df_t = df_pareto_global[df_pareto_global["Ambient Temperature"] == temp]
        # Pegar a cor correspondente à temperatura no colormap
        cor_linha = cores(norm(temp))
        
        plt.plot(
            df_t["Total Water Cost"], 
            df_t["Total IQB"], 
            color=cor_linha, 
            alpha=0.4, 
            linewidth=2, 
            zorder=2
        )

    plt.title("Evolução das Fronteiras de Pareto: 15°C a 30°C", fontsize=15, fontweight='bold')
    plt.xlabel("Total Water Cost (R$)", fontsize=12)
    plt.ylabel("Qualidade Média do Banho (IQB)", fontsize=12)
    
    # Ajustar legenda
    plt.legend(title="T. Ambiente (°C)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Salva o gráfico
    file_path = os.path.join(path_pareto, "fronteira_pareto_global_RdYlBu.png")
    plt.savefig(file_path, dpi=200)
    print(f"Gráfico global de Pareto salvo em: {file_path}")

def plot_evolucao_iqb_global(df_resultados, folder_path, peso_referencia="0.5,0.5"):
    """
    Gera um gráfico de linhas mostrando a evolução do IQB ação a ação
    para todas as temperaturas em um único gráfico.
    """

    # 1. Filtrar pelo peso de referência para não poluir o gráfico
    # Se o peso exato não existir, pegamos o primeiro disponível
    df_plot = df_resultados[df_resultados["Weights"] == peso_referencia].copy()
    if df_plot.empty:
        peso_referencia = df_resultados["Weights"].unique()[0]
        df_plot = df_resultados[df_resultados["Weights"] == peso_referencia].copy()

    # 2. Identificar colunas de IQB (IQB 1, IQB 2, etc)
    colunas_iqb = [c for c in df_resultados.columns if c.startswith("IQB ") and c.split(" ")[1].isdigit()]
    
    # 3. Transformar o DataFrame para o formato longo (tidy data)
    df_long = df_plot.melt(
        id_vars=["Ambient Temperature"],
        value_vars=colunas_iqb,
        var_name="Ação",
        value_name="IQB"
    )
    
    # Converter 'Ação' para número e garantir que Temperatura seja numérica para o gradiente
    df_long["Ação"] = df_long["Ação"].str.replace("IQB ", "").astype(int)
    df_long["Ambient Temperature"] = df_long["Ambient Temperature"].astype(float)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df_long,
        x="Ação",
        y="IQB",
        hue="Ambient Temperature",
        palette="RdYlBu_r",
        marker="o",
        linewidth=2,
        alpha=0.8
    )

    plt.title(f"IQB Evolution by Temperature (Weights: {peso_referencia})", fontsize=16, fontweight='bold')
    plt.xlabel("Action Index (2-minute intervals)", fontsize=15)
    plt.ylabel("Bathing Quality Index (IQB)", fontsize=15)
    plt.ylim(0, 1.05) 
    plt.xticks(range(1, len(colunas_iqb) + 1))
    
    plt.legend(title="T. Amb (°C)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

def plot_global_por_vetor_pesos(df_resultados, folder_path):
    """
    Gera um gráfico global para cada vetor de pesos.
    Filtro cromático: RdYlBu (Azul - Amarelo - Vermelho).
    Sem labels de temperatura nos pontos.
    """
    path_pesos = os.path.join(folder_path, "plots_por_pesos")
    os.makedirs(path_pesos, exist_ok=True)

    vetores_pesos = df_resultados["Weights"].unique()

    for peso in vetores_pesos:
        df_peso = df_resultados[df_resultados["Weights"] == peso].copy()
        df_peso = df_peso.sort_values("Ambient Temperature")

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Linha de trajetória (cinza claro para não competir com as cores)
        plt.plot(
            df_peso["Total Water Cost"], 
            df_peso["Total IQB"], 
            linestyle="-", 
            color="gray", 
            alpha=0.3, 
            zorder=1
        )

        # Scatter plot com gradiente RdYlBu_r (Blue -> Yellow -> Red)
        scatter = plt.scatter(
            df_peso["Total Water Cost"], 
            df_peso["Total IQB"], 
            c=df_peso["Ambient Temperature"], 
            cmap="RdYlBu_r", 
            s=130, 
            edgecolor="black", 
            linewidth=0.8,
            zorder=2
        )

        # Configurações de títulos e eixos
        plt.title(f"Sensibilidade Climática - Pesos: {peso}", fontsize=14, fontweight='bold')
        plt.xlabel("Total Water Cost (R$)", fontsize=12)
        plt.ylabel("Qualidade Média do Banho (IQB)", fontsize=12)
        
        # Barra de cores (Colorbar) para indicar a temperatura
        cbar = plt.colorbar(scatter)
        cbar.set_label("Ambient Temperature (°C)", fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Tratamento do nome do arquivo
        peso_str = str(peso).replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(" ", "").replace(",", "-")
        file_path = os.path.join(path_pesos, f"global_peso_{peso_str}.png")
        
        plt.savefig(file_path, dpi=200)
        plt.close()

    print(f"Gráficos de evolução por peso salvos em: {path_pesos}")
