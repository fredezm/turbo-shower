# Configuração B - Modelo 3

import ray
import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPILSContinuousAction
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
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
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import LinearReward, MORecordEpisodeStatistics


from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from ray.tune.registry import register_env

from controle_temp_saida_model3_configB import simulacao_malha_temperatura
from controle_temp_saida_model3_configB import modelagem_sistema
from controle_temp_saida_model3_configB import modelo_valvula_saida
from controle_temp_saida_model3_configB import calculo_iqb
from controle_temp_saida_model3_configB import custo_eletrico_banho
from controle_temp_saida_model3_configB import custo_gas_banho
from controle_temp_saida_model3_configB import custo_agua_banho

os.environ["WANDB_SILENT"] = "true"

seed = 33
random.seed(seed)
np.random.seed(seed)

# Label para o nome de arquivo de imagens e models  
label_imagens_models = "_teste_50ksteps_30Gradient_Updates_3LayerNetArch512NeuronsEach_gpils"

minutos_banho = 14
if minutos_banho % 2 != 0:
    minutos_banho += 1

# Quantidade total de timesteps
total_timesteps = 50000

class ShowerEnv(gym.Env):
    """Ambiente para simulação do modelo de chuveiro."""

    def __init__(self, **kwargs):

        # Temperatura ambiente, algoritmo, custo da energia elétrica em kWh e modelo:
        self.Tinf_list = kwargs.get("Tinf_list", [25])
        self.nome_algoritmo = kwargs.get("nome_algoritmo", "proximal_policy_optimization")
        self.custo_eletrico_kwh_list = kwargs.get("custo_eletrico_kwh_list", [1])
        
        if "env_config" in kwargs:
            config = kwargs["env_config"]
            self.Tinf = config.get("Tinf", self.Tinf)
            self.nome_algoritmo = config.get("nome_algoritmo", self.nome_algoritmo)
            self.custo_eletrico_kwh = config.get("custo_eletrico_kwh", self.custo_eletrico_kwh)

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

        
        # Ações - SPTs, SPTq, xs, split-range:
    
        if self.nome_algoritmo == "gpi-ls":
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1]),
                shape=(4,),
                dtype=np.float32,
            )   
            self.min_action = np.array([30, 30, 0.01, 0], dtype=np.float32)
            self.max_action = np.array([40, 70, 0.99, 1], dtype=np.float32)

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

                # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf, custo_eletrico_kwh, custo_eletrico, custo_gas, custo_agua:
        
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0]),
            high=np.array([100, 100, 100, 10000, 100, 1, 1, 1, 35, 3, 1, 1, 1]),
            dtype=np.float32, 
        )

        # self.reward_space = gym.spaces.Box(
        #     low=np.array([0,]),
        #     high=np.array([1,]),
        #     shape=(1,),
        #     dtype=np.float32,
        # )

        # self.reward_dim = 1

        # Reward para MO dim 2
        self.reward_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([100, 100]),
            shape=(2,),
            dtype=np.float32,
        )

        self.reward_dim = 2

    def rescale_action(self, action):
        """Converte a ação contínua do GPI para os valores reais do ambiente."""
        # action vem entre -1 e 1, converte para o intervalo real
        scaled_action = self.min_action + (action + 1.0) * 0.5 * (self.max_action - self.min_action)
        return scaled_action

    def reset(self, *, seed=None, options=None):

        # Random seed:
        super().reset(seed=seed)

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

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf, custo_eletrico_kwh, custo_eletrico, custo_gas, custo_agua:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.xq, self.iqb, self.Tinf,
                             self.custo_eletrico_kwh, self.custo_eletrico, self.custo_gas, self.custo_agua],
                             dtype=np.float32)
        
        if self.nome_algoritmo == "gpi-ls":
            self.obs =  (self.obs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        
        return self.obs, {}

    def step(self, action):

        # Tempo de cada iteração:
        self.tempo_final = self.tempo_inicial + self.tempo_iteracao

        if self.nome_algoritmo == "proximal_policy_optimization":
        # Setpoint da temperatura de saída:
            self.SPTs = round(action[0][0], 2)

            # Fração de aquecimento do boiler:
            self.SPTq = round(action[1][0], 1)

            # Abertura da válvula de saída:
            self.xs = round(action[2][0], 2)

            # Split-range:
            self.split_range = action[3]

        if self.nome_algoritmo in ["soft_actor_critic", "gpi-ls"]:
            if self.nome_algoritmo == "gpi-ls":
                action = self.rescale_action(action)

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
        # print("Formato Custo Eletrico:" + self.custo_eletrico)
        # Cálculo do custo de gás do banho:
        self.custo_gas = custo_gas_banho(self.Sa_total, self.potencia_aquecedor, self.custo_gas_kg, self.dt)

        # Cálculo do custo da água:
        self.custo_agua = custo_agua_banho(self.Fs, self.custo_agua_m3, self.tempo_iteracao)

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf, custo_eletrico_kwh, custo_eletrico, custo_gas, custo_agua:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.xq, self.iqb, self.Tinf,
                             self.custo_eletrico_kwh, self.custo_eletrico, self.custo_gas, self.custo_agua],
                             dtype=np.float32)
        
        if self.nome_algoritmo == "gpi-ls":
            self.obs =  (self.obs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)

        custo_total = self.custo_agua + self.custo_eletrico + self.custo_gas
        # Define a recompensa:
        reward = np.array([self.iqb, -self.custo_eletrico], dtype=np.float32)
        # reward = self.iqb

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

        # Termina o episódio se o tempo for maior que (quantidade declarada de minutos) ou se o nível do tanque ultrapassar 100:
        terminated, truncated = False, False
        if self.tempo_final == minutos_banho:
            truncated = True
        if self.h > 100: 
            terminated = True
            reward += - 5.0 

        return self.obs, reward, terminated, truncated, info

    def render(self):
        pass

register(
    id='Shower-v0',
    entry_point='__main__:ShowerEnv',
)

def treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, Tinf_list, custo_eletrico_kwh_list):

    if (nome_algoritmo not in ["gpi-ls", "proximal_policy_optimization", "soft_actor_critic"]):
        raise ValueError("Algoritmo nao suportado")

    # Define o local para salvar o modelo treinado e os checkpoints:
    path_root_models = "/models_v2" + f"/models{label_imagens_models}_model3_configB/"
    path_root = os.getcwd() + path_root_models
    path = path_root + "results_" + nome_algoritmo
    
    # Cria o diretório se não existir
    os.makedirs(path, exist_ok=True)

    # Define as configurações para o algoritmo e constrói o agente:
    if nome_algoritmo != "gpi-ls":
        # Define as configurações para o algoritmo e constrói o agente:
        if nome_algoritmo == "proximal_policy_optimization":
            config = ppo.PPOConfig()

        if nome_algoritmo == "soft_actor_critic":
            config = sac.SACConfig()

        # Constrói o agente:
        config.environment(env=ShowerEnv, env_config={"Tinf_list": Tinf_list, "nome_algoritmo": nome_algoritmo, "custo_eletrico_kwh_list": custo_eletrico_kwh_list})
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

            
    else:
        ref_point = np.array([-0.1, -0.1])

        def make_env(record_episode_stats=True):
            # Cria o ambiente personalizado
            env_config={"Tinf_list": Tinf_list, 
                        "nome_algoritmo": nome_algoritmo,
                        "custo_eletrico_kwh_list": custo_eletrico_kwh_list,}
            env = gym.make("Shower-v0", **env_config)
            if record_episode_stats:
                env = MORecordEpisodeStatistics(env)
            return env
        
        env = make_env(record_episode_stats=True)
        eval_env = make_env(record_episode_stats=False)

        agent = GPILSContinuousAction(
            env=env,
            gamma=0.99,
            learning_rate=3e-4,
            learning_starts=10000,
            gradient_updates=30,
            policy_noise=0.2,
            net_arch=[512, 512, 512],
            project_name="ShowerRL",
            experiment_name=f"gpi_ls_model3_configB",
            use_gpi=False,            
        )

        print("Iniciando treinamento do GPILSContinuousAction...")
        
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            ref_point=ref_point,
            known_pareto_front=None,
            # timesteps_per_iter=1000,
        )
        print("Treinamento do GPILSContinuousAction concluído.")

        # Para o GPILSContinuousAction, define n_iter_agente como 1
        n_iter_agente = 1

        model_path = os.path.join(path, f"gpi_ls_model3_configB.zip")
        agent.save(model_path)
        print(f"Modelo GPILSContinuousAction salvo em: {model_path}")
        
    return path

def carrega_agente(nome_algoritmo, Tinf_list, custo_eletrico_kwh_list):
    os.environ["WANDB_DISABLED"] = "true"
    
    path_root_models = "/models_v2" + f"/models{label_imagens_models}_model3_configB/"
    path_root = os.getcwd() + path_root_models
    path = path_root + "results_" + nome_algoritmo
    model_path = os.path.join(path, f"gpi_ls_model3_configB.zip")
    
    # O caminho do checkpoint é o próprio diretório de resultados,
    # pois é lá que o agent.save() está salvando os arquivos.
        
    
    # GPILSContinuousAction precisa do ambiente multi-objetivo para avaliação
    def make_env(record_episode_stats=True):
        # Cria o ambiente personalizado
        env_config={"Tinf_list": Tinf_list, 
                    "nome_algoritmo": nome_algoritmo,
                    "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
                    }
        env = gym.make("Shower-v0", **env_config)
        if record_episode_stats:
            env = MORecordEpisodeStatistics(env)
        return env
    

    try:
        env = make_env(record_episode_stats=False)
        eval_env = make_env(record_episode_stats=False)

        
        agent = GPILSContinuousAction(
            env=env,
            gamma=0.99,
            learning_rate=3e-4,
            learning_starts=10000,
            gradient_updates=30,
            policy_noise=0.2,
            net_arch=[512, 512, 512],
            project_name="ShowerRL",
            experiment_name=f"gpi_ls_model3_configB",
            use_gpi=False,
        )
        
        print(f"Carregando modelo GPILSContinuousAction de: {model_path}")
        agent.load(model_path + f"/gpi_ls_model3_configB.tar")
        print("Modelo GPILSContinuousAction carregado com sucesso!")
        weights_treinados = agent.weight_support
            

    except Exception as e:
        print(f"ERRO: Falha ao restaurar o checkpoint de '{model_path}'.")
        print(f"Detalhes do erro: {e}")
        print("Verifique o conteúdo do diretório para confirmar se os arquivos de checkpoint estão presentes.")
        return

    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em '{model_path}'")
        return
    
    
    print("Agente restaurado com sucesso!")
    return agent, env

def avalia_agente(nome_algoritmo, Tinf_list, custo_eletrico_kwh_list, agent, env, weights_avaliacao = [0.5, 0.5]):

    # Temperatura ambiente e custo da energia elétrica:
    Tinf_var = str(Tinf_list[0]).replace(".", "-")
    custo_eletrico_kwh_var = str(custo_eletrico_kwh_list[0]).replace(".", "-")
    Tinf_num = Tinf_list[0]
    custo_eletrico_kwh_num = custo_eletrico_kwh_list[0]

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
    concepts_selecionados_list = []

    # Roda o episódio com as ações sugeridas pelo agente treinado:
    i = 1
    terminated, truncated = False, False

    episode_reward = 0
    print(f"Episódio {i}.")

    # Reseta o ambiente
    obs, info = env.reset()

    while i < (1 + minutos_banho/2) and not truncated and not terminated:

        # Seleciona ações:
        if nome_algoritmo == "gpi-ls":            
            w = weights_avaliacao             
            action = agent.eval(obs, w=w)
        else: # PPO, SAC
            action = agent.compute_single_action(obs)

        print(f"Iteração: {i}")
        print(f"Ação: {action}")
        concepts_selecionados_list.append(action)

        # Retorna os estados e a recompensa:
        obs, reward, terminated, truncated, info = env.step(action)
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

        i += 1

        print(f"Recompensa total: {episode_reward}")
        print("")

        tempo_total = np.arange(start=0, stop=2*(i-1) + 0.01*(i-1), step=0.01, dtype="float")    
        tempo_acoes = np.arange(start=1, stop=i, step=1, dtype="int")
        
        # Custos cumulativos:
        custo_eletrico_list_acumulado = list(accumulate(custo_eletrico_list))
        custo_gas_list_acumulado = list(accumulate(custo_gas_list))
        custo_agua_list_acumulado = list(accumulate(custo_agua_list))
        custo_total_list_acumulado = []
        for l in range(len(custo_gas_list_acumulado)):
            custo_total_list_acumulado.append(custo_eletrico_list_acumulado[l] + custo_agua_list_acumulado[l] + custo_gas_list_acumulado[l])

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

        weights_str = ','.join(map(str, weights_avaliacao))

        resultados_list = [
            weights_str,
            Tinf_num, 
            custo_eletrico_kwh_num, 
            *iqb_list,
            IQB_mean,
            IQB_total_sum,
            episode_reward, 
            custo_eletrico_total, 
            custo_gas_total, 
            custo_agua_total,
            custo_total_banho,
        ]

        concepts_list = [
            weights_str,
            Tinf_num, 
            custo_eletrico_kwh_num, 
            *concepts_selecionados_list,
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
    path_imagens = os.getcwd() + f"/imagens" + f"/imagens{label_imagens_models}_model3_configB/" + weights_str + "/Tinf" + Tinf_var + "/"

    # Diretório para salvar as imagens:
    os.makedirs(path_imagens, exist_ok=True)

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
    plt.savefig(path_imagens + "resultado1_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + ".png", dpi=200)
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
    plt.savefig(path_imagens + "resultado2_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + ".png", dpi=200)
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
    ax[2].plot(tempo_acoes, custo_total_list_acumulado, label="Custo total", color="black", linestyle="dashed")
    ax[2].set_title("Custos cumulativos do banho")
    ax[2].set_xlabel("Ação")
    ax[2].set_ylabel("Custos em reais")
    ax[2].legend()
    plt.savefig(path_imagens + "resultado3_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + ".png", dpi=200)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(tempo_acoes, iqb_list, label="IQB", color="black", linestyle="solid")
    ax.set_title("Índice de qualidade do banho (IQB)")
    ax.set_xlabel("Ação")
    ax.set_ylabel("Índice")
    ax.legend()
    plt.savefig(path_imagens + "resultado4_" + nome_algoritmo + "_Tinf" + Tinf_var + "_tarifa" + custo_eletrico_kwh_var + ".png", dpi=200)
    plt.cla()
    plt.close(fig)


    return resultados_list, concepts_list

def plot_fronteira_pareto(df_resultados, folder_path):
    """
    Gera gráficos da Fronteira de Pareto para cada temperatura ambiente avaliada.
    Eixo X: Custo Total (Minimizar)
    Eixo Y: IQB Médio (Maximizar)
    """
    # Garante que o diretório existe
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)

    # Obtém a lista única de temperaturas avaliadas
    temperaturas = df_resultados["Temperatura ambiente"].unique()

    print(f"Gerando gráficos de Pareto em: {path_pareto}")

    for temp in temperaturas:
        # Filtra os dados apenas para aquela temperatura
        df_temp = df_resultados[df_resultados["Temperatura ambiente"] == temp]

        plt.figure(figsize=(10, 6))
        
        # Plotagem dos pontos (Cada ponto é um vetor de pesos diferente)
        # Eixo X: Custo (Quanto mais à esquerda, melhor)
        # Eixo Y: IQB (Quanto mais para cima, melhor)
        sns.scatterplot(
            data=df_temp, 
            x="Custo elétrico total", 
            y="IQB médio", 
            s=100, # Tamanho do ponto
            color="tab:blue",
            edgecolor="black"
        )

        # Adiciona anotações para mostrar qual peso gerou aquele ponto
        # Isso ajuda a entender qual preferência leva a qual resultado
        for _, row in df_temp.iterrows():
            # Formatando o peso para ficar legível no gráfico
            peso_str = row["Pesos"] 
            plt.annotate(
                peso_str, 
                (row["Custo elétrico total"], row["IQB médio"]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7
            )

        plt.title(f"Fronteira de Pareto Aproximada - T. Amb: {temp}°C")
        plt.xlabel("Custo elétrico total (R$)")
        plt.ylabel("IQB Médio")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Salva o gráfico
        temp_str = str(temp).replace(".", "-")
        plt.savefig(os.path.join(path_pareto, f"pareto_T{temp_str}.png"), dpi=150)
        plt.close() # Fecha a figura para liberar memória

    print("Gráficos de Pareto gerados com sucesso.")

def plot_comparativo_pareto_clima(df_resultados, folder_path, temp_fria, temp_amena, temp_quente):
    """
    Gera um gráfico único comparando a Fronteira de Pareto para três climas diferentes.
    """
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)

    # Cores e mapeamento
    climas = {
        temp_fria: {"label": f"Fria ({temp_fria}°C)", "color": "blue"},
        temp_amena: {"label": f"Amena ({temp_amena}°C)", "color": "orange"},
        temp_quente: {"label": f"Quente ({temp_quente}°C)", "color": "red"}
    }

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    for temp, config in climas.items():
        # Filtra os dados para a temperatura específica
        df_temp = df_resultados[df_resultados["Temperatura ambiente"] == temp]
        
        if df_temp.empty:
            print(f"Aviso: Dados para a temperatura {temp}°C não encontrados no DataFrame.")
            continue

        # Ordenar os dados pelo custo para desenhar uma linha de fronteira (opcional)
        df_temp = df_temp.sort_values("Custo elétrico total")

        # Plotar os pontos
        plt.scatter(
            df_temp["Custo elétrico total"], 
            df_temp["IQB médio"], 
            s=120, 
            color=config["color"], 
            label=config["label"],
            edgecolor="black",
            zorder=3
        )

        # Plotar uma linha suave conectando para visualizar a "fronteira"
        plt.plot(
            df_temp["Custo elétrico total"], 
            df_temp["IQB médio"], 
            color=config["color"], 
            linestyle="--", 
            alpha=0.5,
            zorder=2
        )

    plt.title("Comparação de Fronteiras de Pareto por Clima", fontsize=14, fontweight='bold')
    plt.xlabel("Custo elétrico total (R$)", fontsize=12)
    plt.ylabel("IQB Médio (Qualidade)", fontsize=12)
    plt.legend(title="Condição Climática")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    # Salva o gráfico comparativo
    file_path = os.path.join(path_pareto, "comparativo_pareto_climas.png")
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    print(f"Gráfico comparativo salvo em: {file_path}")

def plot_fronteira_pareto_global(df_resultados, folder_path):
    """
    Gera um gráfico único com todas as temperaturas usando um gradiente de cor.
    X: Custo elétrico total | Y: IQB Médio
    """
    path_pareto = os.path.join(folder_path, "pareto_plots")
    os.makedirs(path_pareto, exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Ordenar por temperatura para o gradiente de cor fazer sentido
    df_plot = df_resultados.sort_values(by=["Temperatura ambiente", "Custo elétrico total"])

    # Criar o scatter plot com gradiente (palette coolwarm: azul para frio, vermelho para quente)
    scatter = sns.scatterplot(
        data=df_plot,
        x="Custo elétrico total",
        y="IQB médio",
        hue="Temperatura ambiente",
        palette="coolwarm",
        s=80,
        edgecolor="black",
        alpha=0.8,
        zorder=3
    )

    # Opcional: Desenhar linhas conectando os pontos de mesma temperatura para ver a curva
    for temp in df_plot["Temperatura ambiente"].unique():
        df_temp = df_plot[df_plot["Temperatura ambiente"] == temp]
        # Pegar a cor usada pelo seaborn para essa temperatura
        color = scatter.get_legend().get_lines()[0].get_color() # Aproximação
        plt.plot(
            df_temp["Custo elétrico total"], 
            df_temp["IQB médio"], 
            color="gray", # Linha discreta para não poluir
            linestyle="-", 
            linewidth=1, 
            alpha=0.3,
            zorder=2
        )

    plt.title("Evolução da Fronteira de Pareto: 15°C a 30°C", fontsize=15, fontweight='bold')
    plt.xlabel("Custo Elétrico Total (R$)", fontsize=12)
    plt.ylabel("Qualidade Média do Banho (IQB)", fontsize=12)
    
    # Ajustar legenda para mostrar uma escala de cores
    plt.legend(title="T. Ambiente (°C)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Salva o gráfico
    file_path = os.path.join(path_pareto, "fronteira_pareto_global_gradiente.png")
    plt.savefig(file_path, dpi=200)
    print(f"Gráfico global de Pareto salvo em: {file_path}")

def plot_evolucao_iqb_global(df_resultados, folder_path, peso_referencia="0.5,0.5"):
    """
    Gera um gráfico de linhas mostrando a evolução do IQB ação a ação
    para todas as temperaturas em um único gráfico.
    """

    # 1. Filtrar pelo peso de referência para não poluir o gráfico
    # Se o peso exato não existir, pegamos o primeiro disponível
    df_plot = df_resultados[df_resultados["Pesos"] == peso_referencia].copy()
    if df_plot.empty:
        peso_referencia = df_resultados["Pesos"].unique()[0]
        df_plot = df_resultados[df_resultados["Pesos"] == peso_referencia].copy()

    # 2. Identificar colunas de IQB (IQB 1, IQB 2, etc)
    colunas_iqb = [c for c in df_resultados.columns if c.startswith("IQB ") and c.split(" ")[1].isdigit()]
    
    # 3. Transformar o DataFrame para o formato longo (tidy data)
    df_long = df_plot.melt(
        id_vars=["Temperatura ambiente"],
        value_vars=colunas_iqb,
        var_name="Ação",
        value_name="IQB"
    )
    
    # Converter 'Ação' para número e garantir que Temperatura seja numérica para o gradiente
    df_long["Ação"] = df_long["Ação"].str.replace("IQB ", "").astype(int)
    df_long["Temperatura ambiente"] = df_long["Temperatura ambiente"].astype(float)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Plotar as linhas com gradiente coolwarm (frio=azul, quente=vermelho)
    sns.lineplot(
        data=df_long,
        x="Ação",
        y="IQB",
        hue="Temperatura ambiente",
        palette="coolwarm",
        marker="o",
        linewidth=2,
        alpha=0.8
    )

    plt.title(f"Evolução do IQB por Temperatura (Peso: {peso_referencia})", fontsize=15, fontweight='bold')
    plt.xlabel("Número da Ação (Intervalos de 2 min)", fontsize=12)
    plt.ylabel("Índice de Qualidade do Banho (IQB)", fontsize=12)
    plt.ylim(0, 1.05) 
    plt.xticks(range(1, len(colunas_iqb) + 1))
    
    plt.legend(title="T. Amb (°C)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

if __name__ == "__main__":

    # Argumentos:
    parser = argparse.ArgumentParser()
    parser.add_argument("nome_algoritmo", help="Nome do algoritmo", choices=("ppo", "sac", "gpils"))
    parser.add_argument("treina", help="Treina o agente", choices=("True", "False"))
    parser.add_argument("avalia", help="Avalia o agente", choices=("True", "False"))    
    args = vars(parser.parse_args())
    
    # Define o algoritmo:
    if args["nome_algoritmo"] == "gpils":
        nome_algoritmo = "gpi-ls"
        n_iter_agente = 1 
        n_iter_checkpoints = 1

    # Define a temperatura ambiente e o custo da energia elétrica:
    Tinf_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    # Tinf_list = [25, 26, 27, 28, 29, 30]
    # custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]
    custo_eletrico_kwh_list = [1]
    # Treina o agente:
    # Treina e avalia o agente:
    if args["treina"] == "True":
        treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, Tinf_list, custo_eletrico_kwh_list)

    # Avalia o agente:
    if args["avalia"] == "True":
        
        # Tabelas com resultados principais:    
        cols_fixas_tarifa = ["Pesos", "Temperatura ambiente", "Tarifa da energia Selétrica"]
        cols_iqb_tarifa = [f"IQB {i+1}" for i in range(int(minutos_banho/2))]
        cols_fixas_finais_tarifa = ["IQB médio", "IQB total", "Recompensa total", "Custo elétrico total", "Custo de gás total", "Custo de água total","Custo total do banho"]
        df_resultados = pd.DataFrame(
            columns= cols_fixas_tarifa + cols_iqb_tarifa + cols_fixas_finais_tarifa
        )        

        cols_acoes_tarifa = [f"Concept ação {i+1}" for i in range(int(minutos_banho/2))]
        
        df_concepts = pd.DataFrame(
            columns= cols_fixas_tarifa + cols_acoes_tarifa
        )

        # Cria combinações com todas as temperaturas e tarifa:
        combs = list(itertools.product(map(str, Tinf_list), map(str, custo_eletrico_kwh_list)))
        
        # Chamada apenas para pegar os valores dos pesos treinados
        agent, _ = carrega_agente(nome_algoritmo, [0], [0])
        weights_avaliacao = agent.weight_support

        # weights_avaliacao = [[0.8,0.2],
        #                      [0.6,0.4],
        #                      [0.5,0.5],
        #                      [0.4,0.6],
        #                      [0.2,0.8]]

        # weights_avaliacao = [0.5,0.5]

        for i in range(len(weights_avaliacao)):
            for j, k in combs:
                Tinf_val = float(j)
                custo_eletrico_kwh_val = float(k)
                agent, env = carrega_agente(nome_algoritmo, [Tinf_val], [custo_eletrico_kwh_val])
                resultados_list, concepts_list = avalia_agente(nome_algoritmo, [Tinf_val], [custo_eletrico_kwh_val], agent, env, weights_avaliacao[i].cpu().numpy()) 
                # resultados_list, concepts_list = avalia_agente(nome_algoritmo, [Tinf_val], [custo_eletrico_kwh_val], agent, env, weights_avaliacao[i])
                df_resultados.loc[len(df_resultados)] = resultados_list + [None] * (len(df_resultados.columns) - len(resultados_list))
                df_concepts.loc[len(df_concepts)] = concepts_list + [None] * (len(df_concepts.columns) - len(concepts_list))

        # # Salva os resultados principais em um arquivo csv:
    
        df_resultados.to_csv("./resultados_tabela_programmed/resultados_tabela.csv", index=False)

        print("Iniciando plotagem das Fronteiras de Pareto...")
        path_output = os.getcwd() + f"/imagens" + f"/imagens{label_imagens_models}_model3_configB/" + "pareto_plots/"
        plot_fronteira_pareto(df_resultados, path_output)

        print("Gerando gráfico comparativo de climas...")
        # Escolha as temperaturas desejadas aqui (devem existir em Tinf_list)
        path_output = os.getcwd() + f"/imagens/imagens{label_imagens_models}_model3_configB/"
        plot_comparativo_pareto_clima(
            df_resultados, 
            path_output, 
            temp_fria=17.0, 
            temp_amena=22.0, 
            temp_quente=27.0
        )

        print("Gerando análises globais...")
        path_output = os.getcwd() + f"/imagens/imagens{label_imagens_models}_model3_configB/"
        
        # Chame a nova função aqui
        plot_fronteira_pareto_global(df_resultados, path_output)
        
        pesos_unicos = df_resultados["Pesos"].astype(str).str.replace('"', '').str.strip().unique()
        
        print(f"Detectados {len(pesos_unicos)} pesos diferentes. Gerando gráficos de evolução...")
        
        for peso in pesos_unicos:
            print(f" -> Gerando gráfico para o peso: {peso}")
            plot_evolucao_iqb_global(df_resultados, path_output, peso_referencia=peso)

            path_plot = os.path.join(path_output, "analise_global")
            os.makedirs(path_plot, exist_ok=True)
            unique_weight_name = f"evolucao_iqb_global_temperaturas_peso_{peso}.png"
            file_path = os.path.join(path_plot, unique_weight_name)
            plt.savefig(file_path, dpi=200)
            print(f"Gráfico de evolução do IQB salvo em: {file_path}")