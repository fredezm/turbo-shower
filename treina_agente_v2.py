# Configuração B - Modelo 3

import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy

import argparse
import itertools
import random
import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import accumulate

from controle_temperatura_saida import simulacao_malha_temperatura
from controle_temperatura_saida import modelagem_sistema
from controle_temperatura_saida import modelo_valvula_saida
from controle_temperatura_saida import calculo_iqb
from controle_temperatura_saida import custo_eletrico_banho
from controle_temperatura_saida import custo_gas_banho
from controle_temperatura_saida import custo_agua_banho


class ShowerEnv(gym.Env):
    """Ambiente para simulação do modelo de chuveiro."""

    def __init__(self, env_config):

        # Temperatura ambiente, algoritmo, custo da energia elétrica em kWh, selector e modelo:
        self.Tinf_list = env_config["Tinf_list"]
        self.nome_algoritmo = env_config["nome_algoritmo"]
        self.custo_eletrico_kwh_list = env_config["custo_eletrico_kwh_list"]
        self.selector = env_config["selector"]
        self.model = env_config["model"]

        # Tempo de simulação:
        self.dt = 0.01

        # Tempo de cada iteracao:
        self.tempo_iteracao = 2

        # Utiliza split-range:
        self.Sr = 0

        # Potência da resistência elétrica em kW:
        self.potencia_eletrica = 5.5

        # Potência do aquecedor boiler em kcal/h:
        self.potencia_aquecedor = 29000

        # Custo do kg do gás e do m3 da água:
        self.custo_gas_kg = 3
        self.custo_agua_m3 = 4

        # Concept selector seleciona qual concept treinado será utilizado:
        if self.selector == True:
            self.action_space = gym.spaces.Discrete(3)
            self.models = []

            for i in self.model:
                self.models.append(Policy.from_checkpoint(glob.glob(i+"/*")[-1])['default_policy'])

            self.model = self.models

        # Ações - SPTs, SPTq, xs, split-range:
        else:
            if self.nome_algoritmo == "proximal_policy_optimization":
                self.action_space = gym.spaces.Tuple(
                    (
                        gym.spaces.Box(low=30, high=40, shape=(1,), dtype=np.float32),
                        gym.spaces.Box(low=30, high=70, shape=(1,), dtype=np.float32),
                        gym.spaces.Box(low=0.01, high=0.99, shape=(1,), dtype=np.float32),
                        gym.spaces.Discrete(2, start=0),
                    ),
                )
            
            # SAC não funciona com Tuple space:
            if self.nome_algoritmo == "soft_actor_critic":
                self.action_space = gym.spaces.Box(
                    low=np.array([30, 30, 0.01, 0]), 
                    high=np.array([40, 70, 0.99, 1]), 
                    dtype=np.float32
                )

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf:
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 10]),
            high=np.array([100, 100, 100, 10000, 100, 1, 1, 1, 35]),
            dtype=np.float32, 
        )

    def reset(self):
        
        # Temperatura ambiente e custo da energia elétrica em kWh:
        self.Tinf = random.choice(self.Tinf_list)
        self.custo_eletrico_kwh = random.choice(self.custo_eletrico_kwh_list)

        # Distúrbios Fd e Td, temperatura da corrente fria Tf:
        self.Fd = 0
        self.Td = self.Tinf
        self.Tf = self.Tinf

        # Tempo inicial:
        self.tempo_inicial = 0

        # Nível do tanque de aquecimento e setpoint:
        self.h = 80
        self.SPh = 80

        # Temperatura de saída:
        self.Ts = self.Tinf

        # Temperatura do boiler:
        self.Tq = 55

        # Temperatura do tanque:
        self.Tt = self.Tinf

        # Vazão de saída:
        self.Fs = 0

        # Abertura da válvula quente:
        self.xq = 0

        # Abertura da válvula fria:
        self.xf = 0

        # Índice de qualidade do banho:
        self.iqb = 0

        # Custo elétrico do banho:
        self.custo_eletrico = 0

        # Custo do gás do banho:
        self.custo_gas = 0

        # Custo da água do banho:
        self.custo_agua = 0

        # Condições iniciais - Tq, h, Tt, Ts:
        self.Y0 = np.array([self.Tq, self.h] + 50 * [self.Tinf])

        # Define o buffer para os ganhos integral e derivativo das malhas de controle:
        # 0 - malha boiler, 1 - malha nível, 2 - malha tanque, 3 - malha saída
        id = [0, 1, 2, -1]
        self.Kp = np.array([1, 0.3, 2.0, 0.51])
        self.b = np.array([1, 1, 1, 0.8])
        self.I_buffer = self.Kp * self.Y0[id] * (1 - self.b)
        self.D_buffer = np.array([0, 0, 0, 0])  

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.xq, self.iqb, self.Tinf],
                             dtype=np.float32)
        
        return self.obs

    def step(self, action):

        # Tempo de cada iteração:
        self.tempo_final = self.tempo_inicial + self.tempo_iteracao

        if self.selector == True:
            actions = self.model[action].compute_single_action(self.obs, explore=False)[0]

            # Ambos os algoritmos realizam uma normalização interna antes de o agente selecionar as ações
            # Logo, é preciso reverter essa normalização, pois ela não é feita automaticamente

            if self.nome_algoritmo == "proximal_policy_optimization":
                # Setpoint da temperatura de saída:
                self.SPTs = round((actions[0][0] * np.std([30, 40])) + np.mean([30, 40]), 2)
                if self.SPTs > 40:
                    self.SPTs = 40
                if self.SPTs < 30:
                    self.SPTs = 30

                # Setpoint da temperatura do boiler:
                self.SPTq = round((actions[1][0] * np.std([30, 70])) + np.mean([30, 70]), 1)
                if self.SPTq > 70:
                    self.SPTq = 70
                if self.SPTq < 30:
                    self.SPTq = 30

                # Abertura da válvula de saída:
                self.xs = round((actions[2][0] * np.std([0.01, 0.99])) + np.mean([0.01, 0.99]), 2)
                if self.xs > 0.99:
                    self.xs = 0.99
                if self.xs < 0.01:
                    self.xs = 0.01

                # Split-range:
                self.split_range = actions[3]

            if self.nome_algoritmo == "soft_actor_critic":
                # Setpoint da temperatura de saída:
                self.SPTs = round((action[0] * np.std([30, 40])) + np.mean([30, 40]), 2)
                if self.SPTs > 40:
                    self.SPTs = 40
                if self.SPTs < 30:
                    self.SPTs = 30

                # Fração de aquecimento do boiler:
                self.SPTq = round((action[1] * np.std([30, 70])) + np.mean([30, 70]), 1)
                if self.SPTq > 70:
                    self.SPTq = 70
                if self.SPTq < 30:
                    self.SPTq = 30

                # Abertura da válvula de saída:
                self.xs = round((action[2] * np.std([0.01, 0.99])) + np.mean([0.01, 0.99]), 2)
                if self.xs > 0.99:
                    self.xs = 0.99
                if self.xs < 0.01:
                    self.xs = 0.01

                # Split-range:
                self.split_range = round((action[3] * np.std([0, 1])) + np.mean([0, 1]))      
                if self.split_range > 1:
                    self.split_range = 1
                if self.split_range < 0:
                    self.split_range = 0
        else:
            if self.nome_algoritmo == "proximal_policy_optimization":
                # Setpoint da temperatura de saída:
                self.SPTs = round(action[0][0], 2)

                # Fração de aquecimento do boiler:
                self.SPTq = round(action[1][0], 1)

                # Abertura da válvula de saída:
                self.xs = round(action[2][0], 2)

                # Split-range:
                self.split_range = action[3]

            if self.nome_algoritmo == "soft_actor_critic":
                # Setpoint da temperatura de saída:
                self.SPTs = round(action[0], 2)

                # Fração de aquecimento do boiler:
                self.SPTq = round(action[1], 1)

                # Abertura da válvula de saída:
                self.xs = round(action[2], 2)

                # Split-range:
                self.split_range = round(action[3])    

        # Variáveis para simulação - tempo, SPTq, SPh, xq, xs, Tf, Td, Tinf, Fd, Sr:
        self.UT = np.array(
            [   
                [self.tempo_inicial, self.SPTq, self.SPh, self.SPTs, self.xs, self.Tf, self.Td, self.Tinf, self.Fd, self.Sr],
                [self.tempo_final, self.SPTq, self.SPh, self.SPTs, self.xs, self.Tf, self.Td, self.Tinf, self.Fd, self.Sr]
            ]
        )

        # Solução do sistema:
        self.TT, self.YY, self.UU, self.Y0, self.I_buffer, self.D_buffer = simulacao_malha_temperatura(
            modelagem_sistema, 
            self.Y0, 
            self.UT, 
            self.dt, 
            self.I_buffer,
            self.D_buffer,
            self.Tinf,
            self.split_range
        )

        # Valor final da temperatura do boiler:
        self.Tq = self.YY[:,0][-1]

        # Valor final do nível do tanque:
        self.h = self.YY[:,1][-1]

        # Valor final da temperatura do tanque:
        self.Tt = self.YY[:,2][-1]

        # Valor final da temperatura de saída:
        self.Ts = self.YY[:,3][-1]

        # Fração do aquecedor do boiler utilizada durante a iteração:
        self.Sa_total =  self.UU[:,0]

        # Fração da resistência elétrica utilizada durante a iteração:
        self.Sr_total = self.UU[:,8]

        # Valor final da abertura de corrente fria:
        self.xf = self.UU[:,1][-1]

        # Valor final da abertura de corrente quente:
        self.xq = self.UU[:,2][-1]

        # Valor final da abertura da válvula de saída:
        self.xs = self.UU[:,3][-1]

        # Valor final da vazão de saída:
        self.Fs = modelo_valvula_saida(self.xs)

        # Cálculo do índice de qualidade do banho:
        self.iqb = calculo_iqb(self.Ts, self.Fs)

        # Cálculo do custo elétrico do banho:
        self.custo_eletrico = custo_eletrico_banho(self.Sr_total, self.potencia_eletrica, self.custo_eletrico_kwh, self.dt)

        # Cálculo do custo de gás do banho:
        self.custo_gas = custo_gas_banho(self.Sa_total, self.potencia_aquecedor, self.custo_gas_kg, self.dt)

        # Cálculo do custo da água:
        self.custo_agua = custo_agua_banho(self.Fs, self.custo_agua_m3, self.tempo_iteracao)

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.xq, self.iqb, self.Tinf],
                             dtype=np.float32)

        # Define a recompensa:
        reward = self.iqb

        # Incrementa tempo inicial:
        self.tempo_inicial = self.tempo_inicial + self.tempo_iteracao

        # Para visualização:
        self.SPTq_total = np.repeat(self.SPTq, 201)
        self.Tq_total = self.YY[:,0]
        self.SPh_total = np.repeat(self.SPh, 201)
        self.h_total = self.YY[:,1]
        self.Tt_total = self.YY[:,2]
        self.SPTs_total = np.repeat(self.SPTs, 201)
        self.Ts_total = self.YY[:,3]
        self.xq_total = self.UU[:,2]
        self.xf_total = self.UU[:,1]
        self.xs_total = np.repeat(self.xs, 201)
        self.Fs_total = np.repeat(self.Fs, 201)    
        self.Fd_total = np.repeat(self.Fd, 201) 
        self.Td_total = np.repeat(self.Td, 201) 
        self.Tf_total = np.repeat(self.Tf, 201) 
        self.Tinf_total = np.repeat(self.Tinf, 201) 
        self.split_range_total = np.repeat(self.split_range, 201)

        info = {"SPTq": self.SPTq_total,
                "Tq": self.Tq_total,
                "SPh": self.SPh_total,
                "h": self.h_total,
                "Tt": self.Tt_total,
                "SPTs": self.SPTs_total,
                "Ts": self.Ts_total,
                "Sr": self.Sr_total,
                "Sa": self.Sa_total,
                "xq": self.xq_total,
                "xf": self.xf_total,
                "xs": self.xs_total,
                "Fs": self.Fs_total,
                "iqb": self.iqb,
                "custo_eletrico": self.custo_eletrico,
                "custo_gas": self.custo_gas,
                "custo_agua": self.custo_agua,
                "recompensa": reward,
                "custo_eletrico_kwh": self.custo_eletrico_kwh,
                "Fd": self.Fd_total,
                "Td": self.Td_total,
                "Tf": self.Tf_total,
                "Tinf": self.Tinf_total,
                "split_range": self.split_range_total,}

        # Termina o episódio se o tempo for maior que 14 ou se o nível do tanque ultrapassar 100:
        done = False
        if self.tempo_final == 14 or self.h > 100: 
            done = True

        return self.obs, reward, done, info
    
    def render(self):
        pass


def treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, concept, selector=False, model=None):

    # Define o local para salvar o modelo treinado e os checkpoints:
    path_root_models = "/models/"
    path_root = os.getcwd() + path_root_models
    path = path_root + "results_" + nome_algoritmo + "_concept_" + concept

    # Define os concepts:
    if concept == "banho_dia_frio":
        Tinf_list = [15, 16, 17, 18, 19]
        custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    # if concept == "banho_noite_fria":
    #     Tinf_list = [15, 16, 17, 18, 19]
    #     custo_eletrico_kwh_list = [1.75, 2, 2.25]

    if concept == "banho_dia_ameno":
        Tinf_list = [20, 21, 22, 23, 24]
        custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    # if concept == "banho_noite_amena":
    #     Tinf_list = [20, 21, 22, 23, 24]
    #     custo_eletrico_kwh_list = [1.75, 2, 2.25]

    if concept == "banho_dia_quente":
        Tinf_list = [25, 26, 27, 28, 29, 30]
        custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    # if concept == "banho_noite_quente":
    #     Tinf_list = [25, 26, 27, 28, 29, 30]
    #     custo_eletrico_kwh_list = [1.75, 2, 2.25]

    if concept == "seleciona_banho_v2":
        Tinf_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    # Define as configurações para o algoritmo e constrói o agente:
    if nome_algoritmo == "proximal_policy_optimization":
        config = ppo.PPOConfig()

    if nome_algoritmo == "soft_actor_critic":
        config = sac.SACConfig()

    # Constrói o agente:
    config.environment(env=ShowerEnv, env_config={
        "Tinf_list": Tinf_list, 
        "nome_algoritmo": nome_algoritmo,
        "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
        "selector": selector,
        "model": model
    })
    agent = config.build()

    # Armazena resultados:
    results = []
    episode_data = []

    # Realiza o treinamento:
    for n in range(1, n_iter_agente):

        # Treina o agente:
        result = agent.train()
        results.append(result)
        
        # Armazena dados do episódio:
        episode = {
            "n": n,
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_mean": result["episode_reward_mean"], 
            "episode_reward_max": result["episode_reward_max"],  
            "episode_len_mean": result["episode_len_mean"],
        }
        episode_data.append(episode)

        # Salva checkpoint a cada n_iter_checkpoints iterações:
        if n % n_iter_checkpoints == 0:
            file_name = agent.save(path)
            print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}.')
        else:
            print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}.')
  
    df = pd.DataFrame(data=episode_data)
    df.to_csv(path + "_episode_data" + ".csv")

    return path


def avalia_agente(nome_algoritmo, Tinf_list, custo_eletrico_kwh_list, selector=True):

    # Temperatura ambiente e custo da energia elétrica:
    Tinf_var = str(Tinf_list[0]).replace(".", "-")
    custo_eletrico_kwh_var = str(custo_eletrico_kwh_list[0]).replace(".", "-")
    Tinf_num = Tinf_list[0]
    custo_eletrico_kwh_num = custo_eletrico_kwh_list[0]

    # Define o local do checkpoint salvo:
    path_root_models = "/models/"
    path_root = os.getcwd() + path_root_models
    path = path_root + "results_" + nome_algoritmo + "_concept_"

    banho_dia_frio = path + "banho_dia_frio"
    # banho_noite_fria = path + "banho_noite_fria"
    banho_dia_ameno = path + "banho_dia_ameno"
    # banho_noite_amena = path + "banho_noite_amena"
    banho_dia_quente = path + "banho_dia_quente"
    # banho_noite_quente = path + "banho_noite_quente"
    selector_path = path + "seleciona_banho_v2"

    # model = [banho_dia_frio, banho_noite_fria, banho_dia_ameno, banho_noite_amena, banho_dia_quente, banho_noite_quente]
    model = [banho_dia_frio, banho_dia_ameno, banho_dia_quente]

    # Constrói o ambiente:
    env = ShowerEnv({
        "Tinf_list": Tinf_list, 
        "nome_algoritmo": nome_algoritmo,
        "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
        "selector": selector,
        "model": model
    })
    obs = env.reset()

    # Define se será utilizado o concept selector ou programmed:
    if selector == True:
        agent = Algorithm.from_checkpoint(glob.glob(selector_path +"/*")[-1])
        
    # Define se será utilizado o concept selector ou programmed:
    if selector == True:
        agent = Algorithm.from_checkpoint(glob.glob(selector_path +"/*")[-1])

    if selector == False:
        if Tinf_num < 20: 
            agent = Algorithm.from_checkpoint(glob.glob(banho_dia_frio +"/*")[-1])
        elif Tinf_num >= 20 and Tinf_num < 25: 
            agent = Algorithm.from_checkpoint(glob.glob(banho_dia_ameno +"/*")[-1])
        elif Tinf_num >= 25: 
            agent = Algorithm.from_checkpoint(glob.glob(banho_dia_quente +"/*")[-1])

    # Para visualização:
    SPTq_list = []
    Tq_list = []
    SPh_list = []
    h_list = []
    Tt_list = []
    SPTs_list = []
    Ts_list = []
    split_range_list = []
    Sr_list = []
    Sa_list = []
    xq_list = []
    xf_list = []
    xs_list = []
    Fs_list = []
    iqb_list = []
    custo_eletrico_list = []
    custo_gas_list = []
    custo_agua_list = []
    recompensa_list = []
    Fd_list = []
    Td_list = []
    Tf_list = []
    Tinf_list = []
    tempo_total = np.arange(start=0, stop=14 + 0.07, step=0.01, dtype="float")
    tempo_acoes = np.arange(start=1, stop=8, step=1, dtype="int")
    concepts_selecionados_list = []

    # Roda o episódio com as ações sugeridas pelo agente treinado:
    for i in range(0, 1):

        episode_reward = 0
        print(f"Episódio {i}.")

        for i in range(1, 8):

            # Seleciona ações:
            action = agent.compute_single_action(obs)
            print(f"Iteração: {i}")
            print(f"Ação: {action}")
            concepts_selecionados_list.append(action)

            # Retorna os estados e a recompensa:
            obs, reward, done, info = env.step(action)
            print(f"Estados: {obs}")
            print(f"Temperatura ambiente: {np.unique(info.get('Tinf'))[0]}")
            print(f"Custo elétrico do kWh: {info.get('custo_eletrico_kwh')}")

            # Recompensa total:
            episode_reward += reward
            print(f"Recompensa: {reward}.")
            print("")

            # Para visualização:
            SPTq_list.append(info.get("SPTq"))
            Tq_list.append(info.get("Tq"))
            SPh_list.append(info.get("SPh"))
            h_list.append(info.get("h"))
            Tt_list.append(info.get("Tt"))
            SPTs_list.append(info.get("SPTs"))
            Ts_list.append(info.get("Ts"))
            Sr_list.append(info.get("Sr"))
            Sa_list.append(info.get("Sa"))
            xq_list.append(info.get("xq"))
            xf_list.append(info.get("xf"))
            xs_list.append(info.get("xs"))
            Fs_list.append(info.get("Fs"))
            iqb_list.append(info.get("iqb"))
            custo_eletrico_list.append(info.get("custo_eletrico"))
            custo_gas_list.append(info.get("custo_gas"))
            custo_agua_list.append(info.get("custo_agua"))
            recompensa_list.append(info.get("recompensa"))
            Fd_list.append(info.get("Fd"))
            Td_list.append(info.get("Td"))
            Tf_list.append(info.get("Tf"))
            Tinf_list.append(info.get("Tinf"))
            split_range_list.append(info.get("split_range"))

        print(f"Recompensa total: {episode_reward}")
        print("")

    # Custos cumulativos:
    custo_eletrico_list_acumulado = list(accumulate(custo_eletrico_list))
    custo_gas_list_acumulado = list(accumulate(custo_gas_list))
    custo_agua_list_acumulado = list(accumulate(custo_agua_list))

    # Custos totais:
    custo_eletrico_total = custo_eletrico_list_acumulado[-1]
    custo_gas_total = custo_gas_list_acumulado[-1]
    custo_agua_total = custo_agua_list_acumulado[-1]
    custo_total_banho = custo_eletrico_total + custo_gas_total + custo_agua_total
    print(f"Custo elétrico total: {custo_eletrico_total}")
    print(f"Custo de gás total: {custo_gas_total}")
    print(f"Custo de água total: {custo_agua_total}")
    print(f"Custo total do banho: {custo_total_banho}")

    # Tabelas com resultados principais:
    IQB_total_sum = sum(iqb_list)
    IQB_mean = IQB_total_sum / len(iqb_list)

    resultados_list = [
        Tinf_num, 
        custo_eletrico_kwh_num, 
        iqb_list[0], 
        iqb_list[1], 
        iqb_list[2], 
        iqb_list[3], 
        iqb_list[4], 
        iqb_list[5], 
        iqb_list[6],
        IQB_mean,
        IQB_total_sum,
        episode_reward, 
        custo_eletrico_total, 
        custo_gas_total, 
        custo_agua_total,
        custo_total_banho,
    ]

    concepts_list = [
        Tinf_num, 
        custo_eletrico_kwh_num, 
        concepts_selecionados_list[0], 
        concepts_selecionados_list[1], 
        concepts_selecionados_list[2], 
        concepts_selecionados_list[3],
        concepts_selecionados_list[4], 
        concepts_selecionados_list[5], 
        concepts_selecionados_list[6]
    ]

    # Para visualização:
    SPTq = np.concatenate(SPTq_list, axis=0)
    Tq = np.concatenate(Tq_list, axis=0)
    SPh = np.concatenate(SPh_list, axis=0)
    h = np.concatenate(h_list, axis=0)
    Tt = np.concatenate(Tt_list, axis=0)
    SPTs = np.concatenate(SPTs_list, axis=0)
    Ts = np.concatenate(Ts_list, axis=0)
    Sr = np.concatenate(Sr_list, axis=0)
    Sa = np.concatenate(Sa_list, axis=0)
    xq = np.concatenate(xq_list, axis=0)
    xf = np.concatenate(xf_list, axis=0)
    xs = np.concatenate(xs_list, axis=0)
    Fs = np.concatenate(Fs_list, axis=0)
    Fd = np.concatenate(Fd_list, axis=0)
    Td = np.concatenate(Td_list, axis=0)
    Tf = np.concatenate(Tf_list, axis=0)
    Tinf = np.concatenate(Tinf_list, axis=0)
    split_range = np.concatenate(split_range_list, axis=0)

    # Gráficos:
    sns.set_style("darkgrid")
    path_imagens = os.getcwd() + "/imagens/"

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(tempo_total, Ts, label="Ts", color="tab:blue", linestyle="solid")
    ax[0].plot(tempo_total, Tt, label="Tt", color="tab:red", linestyle="solid")
    ax[0].plot(tempo_total, SPTs, label="SPTs - ação", color="black", linestyle="dashed")
    ax[0].set_title("Setpoint da temperatura de saída (SPTs) e\n temperaturas de saída (Ts) e do tanque (Tt)")
    ax[0].set_xlabel("Tempo em minutos")
    ax[0].set_ylabel("Temperatura em °C")
    ax[0].legend()

    ax[1].plot(tempo_total, Fs, label="Fs", color="tab:red", linestyle="solid")
    ax[1].set_title("Vazão de saída (Fs)")
    ax[1].set_xlabel("Tempo em minutos")
    ax[1].set_ylabel("Vazão em litros/minutos")
    ax[1].legend()

    ax[2].plot(tempo_acoes, iqb_list, label="IQB", color="black", linestyle="solid")
    ax[2].set_title("Índice de qualidade do banho (IQB)")
    ax[2].set_xlabel("Ação")
    ax[2].set_ylabel("Índice")
    ax[2].legend()
    plt.savefig(path_imagens + "resultado1_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + "_selector_" + str(selector) + ".png", dpi=200)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    ax[0, 0].plot(tempo_total, Tq, label="Tq", color="tab:orange", linestyle="solid")
    ax[0, 0].plot(tempo_total, SPTq, label="SPTq - ação", color="black", linestyle="dashed")
    ax[0, 0].set_title("Setpoint da temperatura do boiler (SPTq)\n e temperatura do boiler (Tq)")
    ax[0, 0].set_ylabel("Temperatura °C")
    ax[0, 0].legend()

    ax[0, 1].plot(tempo_total, Sa, label="Sa", color="silver", linestyle="solid")
    ax[0, 1].plot(tempo_total, Sr, label="Sr", color="tab:red", linestyle="solid")
    ax[0, 1].plot(tempo_total, split_range, label="split-range - ação", color="black", linestyle="solid")
    ax[0, 1].set_title("Frações de aquecimento do boiler (Sa)\n e da resistência elétrica (Sr)")
    ax[0, 1].set_ylabel("Fração")
    ax[0, 1].legend()

    ax[1, 0].plot(tempo_total, xs, label="xs - ação", color="black", linestyle="solid")
    ax[1, 0].plot(tempo_total, xq, label="xq", color="tab:red", linestyle="solid")
    ax[1, 0].plot(tempo_total, xf, label="xf", color="tab:blue", linestyle="solid")
    ax[1, 0].set_title("Aberturas das válvulas de saída (xs),\n quente (xq) e fria (xf)")
    ax[1, 0].set_xlabel("Tempo em minutos")
    ax[1, 0].set_ylabel("Abertura")
    ax[1, 0].legend()

    ax[1, 1].plot(tempo_total, SPh, label="SPh", color="black", linestyle="dashed")
    ax[1, 1].plot(tempo_total, h, label="h", color="tab:red", linestyle="solid")
    ax[1, 1].set_title("Setpoint do nível do tanque (SPh) e nível do tanque (h)")
    ax[1, 1].set_xlabel("Tempo em minutos")
    ax[1, 1].set_ylabel("Nível")
    ax[1, 1].legend()
    plt.savefig(path_imagens + "resultado2_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + "_selector_" + str(selector) + ".png", dpi=200)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(20, 4))
    ax[0].plot(tempo_acoes, recompensa_list, label="Recompensa", color="black", linestyle="solid")
    ax[0].set_title("Recompensa do agente")
    ax[0].set_xlabel("Ação")
    ax[0].set_ylabel("Índice")
    ax[0].legend()

    ax[1].plot(tempo_acoes, custo_eletrico_list, label="Custo elétrico", color="tab:blue", linestyle="solid")
    ax[1].plot(tempo_acoes, custo_gas_list, label="Custo do gás", color="tab:red", linestyle="solid")
    ax[1].plot(tempo_acoes, custo_agua_list, label="Custo da água", color="tab:orange", linestyle="solid")
    ax[1].set_title("Custos do banho em cada ação")
    ax[1].set_xlabel("Ação")
    ax[1].set_ylabel("Custos em reais")
    ax[1].legend()

    ax[2].plot(tempo_acoes, custo_eletrico_list_acumulado, label="Custo elétrico", color="tab:blue", linestyle="solid")
    ax[2].plot(tempo_acoes, custo_gas_list_acumulado, label="Custo do gás", color="tab:red", linestyle="solid")
    ax[2].plot(tempo_acoes, custo_agua_list_acumulado, label="Custo da água", color="tab:orange", linestyle="solid")
    ax[2].set_title("Custos cumulativos do banho")
    ax[2].set_xlabel("Ação")
    ax[2].set_ylabel("Custos em reais")
    ax[2].legend()
    plt.savefig(path_imagens + "resultado3_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + "_selector_" + str(selector) + ".png", dpi=200)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(tempo_acoes, iqb_list, label="IQB", color="black", linestyle="solid")
    ax.set_title("Índice de qualidade do banho (IQB)")
    ax.set_xlabel("Ação")
    ax.set_ylabel("Índice")
    ax.legend()
    plt.savefig(path_imagens + "resultado4_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + "_selector_" + str(selector) + ".png", dpi=200)
    plt.cla()
    plt.close(fig)

    return resultados_list, concepts_list


if __name__ == "__main__":

    # Argumentos:
    parser = argparse.ArgumentParser()
    parser.add_argument("nome_algoritmo", help="Nome do algoritmo", choices=("ppo", "sac"))
    parser.add_argument("treina", help="Treina o agente", choices=("True", "False"))
    parser.add_argument("avalia", help="Avalia o agente", choices=("True", "False"))
    parser.add_argument("selector", help="Avalia o agente", choices=("True", "False"))
    args = vars(parser.parse_args())

    # Inicializa o Ray:
    ray.shutdown()
    ray.init()

    # Define o algoritmo:
    if args["nome_algoritmo"] == "ppo":
        nome_algoritmo = "proximal_policy_optimization"
        n_iter_agente = 101
        n_iter_checkpoints = 10

    if args["nome_algoritmo"] == "sac":
        nome_algoritmo = "soft_actor_critic"
        n_iter_agente = 1001
        n_iter_checkpoints = 100

    # Define a temperatura ambiente e o custo da energia elétrica:
    Tinf_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    # Treina o agente:
    if args["treina"] == "True":
        # Treina cada concept:
        banho_dia_frio = treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, "banho_dia_frio")
        # banho_noite_fria = treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, "banho_noite_fria")
        banho_dia_ameno = treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, "banho_dia_ameno")
        # banho_noite_amena = treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, "banho_noite_amena")
        banho_dia_quente = treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, "banho_dia_quente")
        # banho_noite_quente = treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, "banho_noite_quente")

        # model = [banho_dia_frio, banho_noite_fria, banho_dia_ameno, banho_noite_amena, banho_dia_quente, banho_noite_quente]
        model = [banho_dia_frio, banho_dia_ameno, banho_dia_quente]

        # Treina o selector:
        selector = treina_agente(nome_algoritmo, 
            n_iter_agente, 
            n_iter_checkpoints,
            "seleciona_banho_v2", 
            True, 
            model)

    # Avalia o agente:
    if args["avalia"] == "True":
        # Define se será utilizado o concept selector ou programmed:
        if args["selector"] == "True":
            selector = True
        else:
            selector = False

        # Tabelas com resultados principais:    
        df_resultados = pd.DataFrame(
            columns=[
                "Temperatura ambiente", 
                "Tarifa da energia elétrica", 
                "IQB 1", 
                "IQB 2", 
                "IQB 3", 
                "IQB 4", 
                "IQB 5", 
                "IQB 6", 
                "IQB 7", 
                "IQB médio", 
                "IQB total", 
                "Recompensa total", 
                "Custo elétrico total", 
                "Custo de gás total", 
                "Custo de água total",
                "Custo total do banho"
            ]
        )
        df_concepts = pd.DataFrame(
            columns=[
                "Temperatura ambiente", 
                "Tarifa da energia elétrica", 
                "Concept ação 1", 
                "Concept ação 2", 
                "Concept ação 3", 
                "Concept ação 4", 
                "Concept ação 5", 
                "Concept ação 6", 
                "Concept ação 7"
            ]
        )

        # Cria combinações com todas as temperaturas e tarifa:
        combs = list(itertools.product(map(str, Tinf_list), map(str, custo_eletrico_kwh_list)))
        for j, k in combs:
            Tinf_val = float(j)
            custo_eletrico_kwh_val = float(k)
            resultados_list, concepts_list = avalia_agente(nome_algoritmo, [Tinf_val], [custo_eletrico_kwh_val], selector)
            df_resultados.loc[len(df_resultados)] = resultados_list
            df_concepts.loc[len(df_concepts)] = concepts_list

        # Salva os resultados principais em um arquivo csv:
        if selector:
            df_resultados.to_csv("./resultados_tabela_selector/resultados_tabela_Tinf" + str(Tinf_val) + ".csv", index=False)
            df_concepts.to_csv("./resultados_concepts_selector/resultados_concepts_Tinf" + str(Tinf_val) + ".csv", index=False)
        else:
            df_resultados.to_csv("./resultados_tabela_programmed/resultados_tabela.csv", index=False)

    # Reseta o Ray:
    ray.shutdown()