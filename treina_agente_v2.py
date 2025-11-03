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
label_imagens_models = "_teste_SR_gpils"

minutos_banho = 14
if minutos_banho % 2 != 0:
    minutos_banho += 1

# Quantidade total de timesteps
total_timesteps = 500

class ShowerEnv(gym.Env):
    """Ambiente para simulação do modelo de chuveiro."""

    def __init__(self, **kwargs):

        # Temperatura ambiente, algoritmo, custo da energia elétrica em kWh, selector e modelo:
        self.Tinf_list = kwargs.get("Tinf_list", [25])
        self.nome_algoritmo = kwargs.get("nome_algoritmo", "proximal_policy_optimization")
        self.custo_eletrico_kwh_list = kwargs.get("custo_eletrico_kwh_list", [2])
        self.selector = kwargs.get("selector", False)
        self.model = kwargs.get("model", [])

        if "env_config" in kwargs:
            config = kwargs["env_config"]
            self.Tinf = config.get("Tinf", self.Tinf)
            self.nome_algoritmo = config.get("nome_algoritmo", self.nome_algoritmo)
            self.custo_eletrico_kwh = config.get("custo_eletrico_kwh", self.custo_eletrico_kwh)
            self.selector = config.get("selector", self.selector)
            self.model = config.get("model", self.model)

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

        # Ações - SPTs, SPTq, xs, split-range:
        else:
            if self.nome_algoritmo == "gpi-ls":
                self.action_space = gym.spaces.Box(
                    low=np.array([-1, -1, -1, -1]),
                    high=np.array([1, 1, 1, 1]),
                    shape=(4,),
                    dtype=np.float32,
                )   
                self.min_action = np.array([30, 30, 0.01, 0], dtype=np.float32)
                self.max_action = np.array([40, 70, 0.99, 1], dtype=np.float32)

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf:
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 10]),
            high=np.array([100, 100, 100, 10000, 100, 1, 1, 1, 35]),
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

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.xq, self.iqb, self.Tinf],
                             dtype=np.float32)
        
        if self.nome_algoritmo == "gpi-ls":
            self.obs =  (self.obs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        
        return self.obs, {}

    def step(self, action):

        # Tempo de cada iteração:
        self.tempo_final = self.tempo_inicial + self.tempo_iteracao

        if self.selector == True:
            # Para fins de debug, os pesos serão fixos, mas como selecioná-los é um problema que precisará de solução
            weights = [1,0]
            actions = self.model[action].eval(self.obs, weights)

            # Ambos os algoritmos realizam uma normalização interna antes de o agente selecionar as ações
            # Logo, é preciso reverter essa normalização, pois ela não é feita automaticamente

            
            if self.nome_algoritmo in ["soft_actor_critic", "gpi-ls"]:
                # if self.nome_algoritmo == "gpi-ls":
                #     action = self.rescale_action(action)

                # Setpoint da temperatura de saída:
                self.SPTs = round((actions[0] * np.std([30, 40])) + np.mean([30, 40]), 2)
                if self.SPTs > 40:
                    self.SPTs = 40
                if self.SPTs < 30:
                    self.SPTs = 30

                # Fração de aquecimento do boiler:
                self.SPTq = round((actions[1] * np.std([30, 70])) + np.mean([30, 70]), 1)
                if self.SPTq > 70:
                    self.SPTq = 70
                if self.SPTq < 30:
                    self.SPTq = 30

                # Abertura da válvula de saída:
                self.xs = round((actions[2] * np.std([0.01, 0.99])) + np.mean([0.01, 0.99]), 2)
                if self.xs > 0.99:
                    self.xs = 0.99
                if self.xs < 0.01:
                    self.xs = 0.01

                # Split-range:
                self.split_range = round((actions[3] * np.std([0, 1])) + np.mean([0, 1]))      
                if self.split_range > 1:
                    self.split_range = 1
                if self.split_range < 0:
                    self.split_range = 0
        else:
            
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

        # Estados - Ts, Tq, Tt, h, Fs, xf, xq, iqb, Tinf:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.xq, self.iqb, self.Tinf],
                             dtype=np.float32)
        
        if self.nome_algoritmo == "gpi-ls":
            self.obs =  (self.obs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)

        custo_total = self.custo_agua + self.custo_eletrico + self.custo_gas
        # Define a recompensa:
        reward = np.array([self.iqb, -custo_total], dtype=np.float32)
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

def create_shower_env_with_linear_reward(env_config):
    """Cria o ambiente base e aplica o wrapper LinearReward."""
    
    # Cria o ambiente ShowerEnv
    env = ShowerEnv(**env_config)
    
    # Define os pesos [peso_temperatura, peso_vazao]
    weights = np.array([0.8, 0.2])
    
    # Aplica o wrapper LinearReward do mo_gymnasium
    return LinearReward(env, weight=weights)

# Registra esta função com um nome para o Ray usar
register_env("shower_linear_reward_env", create_shower_env_with_linear_reward)

def treina_agente(nome_algoritmo, n_iter_agente, n_iter_checkpoints, concept, selector=False, model=None):

    # Define o local para salvar o modelo treinado e os checkpoints:
    path_root_models = "models" + f"/models{label_imagens_models}_model3_configB/"
    path_root = os.path.join(os.getcwd(), path_root_models)
    path_concepts = os.path.join(path_root, f"results_{nome_algoritmo}")
    path = os.path.join(path_concepts, f"concept_{concept}")

    if(not selector):
        return path
    # Cria o diretório se não existir
    os.makedirs(path, exist_ok=True)

    # Define os concepts:
    if concept == "banho_dia_frio":
        Tinf_list = [15, 16, 17, 18, 19]
        custo_eletrico_kwh_list = [1]
        # custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    if concept == "banho_dia_ameno":
        Tinf_list = [20, 21, 22, 23, 24]
        custo_eletrico_kwh_list = [1]
        # custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    if concept == "banho_dia_quente":
        Tinf_list = [25, 26, 27, 28, 29, 30]
        custo_eletrico_kwh_list = [1]
        # custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    if concept == "seleciona_banho_v2":
        Tinf_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        custo_eletrico_kwh_list = [1]
        # custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]

    # Define as configurações para o algoritmo e constrói o agente:
    if nome_algoritmo != "gpi-ls":    
        raise ValueError("Algoritmo nao suportado")
    
    ref_point = np.array([-0.1, -0.1])

    def make_env(record_episode_stats=True):
        # Cria o ambiente personalizado
        env_config={"Tinf_list": Tinf_list, 
                    "nome_algoritmo": nome_algoritmo,
                    "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
                    "selector": selector,
                    "model": model}
        env = gym.make("Shower-v0", **env_config)
        if record_episode_stats:
            env = MORecordEpisodeStatistics(env)
        return env
    
    if selector:
        print("Carregando os modelos 'concept' para o seletor")
        loaded_models = []

        for concept_path in model:                        
            model_zip_path = os.path.join(concept_path, "gpi_ls_model3_configB.zip")
        
            if not os.path.exists(model_zip_path):
                raise FileNotFoundError(f"Arquivo .zip não encontrado em {model_zip_path}")
            
            temp_env_config = {
                "Tinf_list": [25], # Valor arbitrário
                "nome_algoritmo": nome_algoritmo,
                "custo_eletrico_kwh_list": [1],
                "selector": False, # Importante!
                "model": []
            }

            temp_env = gym.make("Shower-v0", **temp_env_config)

            print(f"Carregado {model_zip_path}...")
        
            agent_to_load = GPILSContinuousAction(
                env = temp_env,
                gamma=0.99,
                learning_rate=3e-4,
                learning_starts=1000,
                gradient_updates=50,
                policy_noise=0.2,
                net_arch=[256, 256, 256],
                project_name="ShowerRL",
                experiment_name=f"gpi_ls_model3_configB",
                use_gpi=False,
            )
            agent_to_load.load(model_zip_path + f"/gpi_ls_model3_configB.tar")

            loaded_models.append(agent_to_load)
            temp_env.close()
        
        print("Todos os concepts foram carregados")

        def make_selector_env(record_episode_stats=True):
            env_config={
                "Tinf_list": Tinf_list,
                "nome_algoritmo": nome_algoritmo,
                "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
                "selector": True,
                "model": loaded_models
            }
            env = gym.make("Shower-v0", **env_config)
            if record_episode_stats:
                env = MORecordEpisodeStatistics(env)
            return env
        env = make_selector_env()
        eval_env = make_selector_env(record_episode_stats=False)

        agent = GPILS(
                env=env,
                gamma=0.99,
                learning_rate=3e-4,
                learning_starts=1000,
                gradient_updates=50,
                net_arch=[256, 256, 256],
                project_name="ShowerRL",
                experiment_name=f"gpi_ls_model3_configB",
                use_gpi=False,
            )
        
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            ref_point=ref_point,
            known_pareto_front=None,
            timesteps_per_iter=100 # Para fins de debug, tirar quando o salvamento estiver ok
        )

        print("Treinamento do GPILSContinuousAction concluído.")

        # Para o GPILSContinuousAction, define n_iter_agente como 1
        model_path = os.path.join(path, f"gpi_ls_model3_configB.zip")        
        agent.save(path)
        print(f"Modelo GPILSContinuousAction salvo em: {path}")
        
    else:
        
        def make_env(record_episode_stats=True):
            # Cria o ambiente personalizado
            env_config={"Tinf_list": Tinf_list, 
                        "nome_algoritmo": nome_algoritmo,
                        "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
                        "selector": selector,
                        "model": model}
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
            learning_starts=1000,
            gradient_updates=50,
            policy_noise=0.2,
            net_arch=[256, 256, 256],
            project_name="ShowerRL",
            experiment_name=f"gpi_ls_model3_configB",
            use_gpi=False,
        )

        print("Iniciando treinamento do GPILSContinuousAction...")
        
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            ref_point=ref_point,
            known_pareto_front=None
        )
        print("Treinamento do GPILSContinuousAction concluído.")

        # Para o GPILSContinuousAction, define n_iter_agente como 1
        n_iter_agente = 1

        model_path = os.path.join(path, f"gpi_ls_model3_configB.zip")
        agent.save(model_path)
        print(f"Modelo GPILSContinuousAction salvo em: {model_path}")

    
    
    # Armazena resultados:
    results = []
    episode_data = []
    
    df = pd.DataFrame(data=episode_data)
    df.to_csv(path + "_episode_data" + ".csv")

    return path

def get_latest_checkpoint(model_path):

    if not os.path.exists(model_path):
        print(f"Diretório não existe: {model_path}")
        return None
    
    if not os.path.isdir(model_path):
        print(f"Caminho não é um diretório: {model_path}")
        return None
    
    try:
        # Listar todos os itens no diretório
        items = os.listdir(model_path)
        
        # Filtrar e ordenar checkpoints
        checkpoints = []
        
        for item in items:
            item_path = os.path.join(model_path, item)
            
            # Verificar se é um diretório
            if os.path.isdir(item_path):
                # RLlib checkpoints podem ter diferentes formatos:
                # - checkpoint_000001
                # - números simples: 1, 2, 3...
                if item.startswith("checkpoint_"):
                    try:
                        num = int(item.split("_")[1])
                        checkpoints.append((num, item_path))
                    except (IndexError, ValueError):
                        pass
                elif item.isdigit():
                    checkpoints.append((int(item), item_path))
        
        # Se não encontrou checkpoints numerados, pegar o mais recente por data
        if not checkpoints:
            dirs_with_time = []
            for item in items:
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path):
                    mtime = os.path.getmtime(item_path)
                    dirs_with_time.append((mtime, item_path))
            
            if dirs_with_time:
                dirs_with_time.sort(key=lambda x: x[0])
                return dirs_with_time[-1][1]
        else:
            # Ordenar por número e retornar o maior
            checkpoints.sort(key=lambda x: x[0])
            return checkpoints[-1][1]
    
    except OSError as e:
        print(f"Erro ao acessar diretório {model_path}: {e}")
    
    return None

def avalia_agente(nome_algoritmo, Tinf_list, custo_eletrico_kwh_list, selector=True, weights_avaliacao = [0.5, 0.5]):

    # Temperatura ambiente e custo da energia elétrica:
    Tinf_var = str(Tinf_list[0]).replace(".", "-")
    custo_eletrico_kwh_var = str(custo_eletrico_kwh_list[0]).replace(".", "-")
    Tinf_num = Tinf_list[0]
    custo_eletrico_kwh_num = custo_eletrico_kwh_list[0]


    path_root_models = "models" + f"/models{label_imagens_models}_model3_configB/"
    path_root = os.path.join(os.getcwd(), path_root_models)
    
    # O caminho do checkpoint é o próprio diretório de resultados,
    # pois é lá que o agent.save() está salvando os arquivos.
    
    if nome_algoritmo == "gpi-ls":
        checkpoint_path = path_root + "results_" + nome_algoritmo

        banho_dia_frio = checkpoint_path + f"/concept_banho_dia_frio"
        # banho_noite_fria = path + "banho_noite_fria"
        banho_dia_ameno = checkpoint_path + f"/concept_banho_dia_ameno"
        # banho_noite_amena = path + "banho_noite_amena"
        banho_dia_quente = checkpoint_path + f"/concept_banho_dia_quente"
        # banho_noite_quente = path + "banho_noite_quente"
        selector_path = checkpoint_path + f"/concept_seleciona_banho_v2"
        if not os.path.isdir(checkpoint_path):
            print(f"ERRO: O diretório de resultados não foi encontrado em '{checkpoint_path}'")
            print("Por favor, execute o treinamento primeiro ('... True False') para criar este diretório e o checkpoint.")
            return
    else:
        checkpoint_path = path_root + "results_" + nome_algoritmo

        banho_dia_frio = checkpoint_path + f"/concept_banho_dia_frio"
        # banho_noite_fria = path + "banho_noite_fria"
        banho_dia_ameno = checkpoint_path + f"/concept_banho_dia_ameno"
        # banho_noite_amena = path + "banho_noite_amena"
        banho_dia_quente = checkpoint_path + f"/concept_banho_dia_quente"
        # banho_noite_quente = path + "banho_noite_quente"
        selector_path = checkpoint_path + f"/concept_seleciona_banho_v2"

    # model = [banho_dia_frio, banho_noite_fria, banho_dia_ameno, banho_noite_amena, banho_dia_quente, banho_noite_quente]
    model = [banho_dia_frio, banho_dia_ameno, banho_dia_quente]

    print(f"Tentando restaurar agente do checkpoint no diretório: {path_root}")

    
    if nome_algoritmo != "gpi-ls":
        raise ValueError("Algoritmo nao suportado")

    # GPILSContinuousAction precisa do ambiente multi-objetivo para avaliação
    def make_env(record_episode_stats=True):
        # Cria o ambiente personalizado
        env_config={"Tinf_list": Tinf_list, 
                    "nome_algoritmo": nome_algoritmo,
                    "custo_eletrico_kwh_list": custo_eletrico_kwh_list,
                    "selector": selector,
                    "model": model}
        env = gym.make("Shower-v0", **env_config)
        if record_episode_stats:
            env = MORecordEpisodeStatistics(env)
        return env
    

    try:
        # Define se será utilizado o concept selector ou programmed:
        env = make_env(record_episode_stats=False)
        eval_env = make_env(record_episode_stats=False)

        if selector == False:
            if Tinf_num < 20: 
                model_path = os.path.join(banho_dia_frio, f"gpi_ls_model3_configB.zip")

            elif Tinf_num >= 20 and Tinf_num < 25: 
                model_path = os.path.join(banho_dia_ameno, f"gpi_ls_model3_configB.zip")

            elif Tinf_num >= 25: 
                model_path = os.path.join(banho_dia_quente, f"gpi_ls_model3_configB.zip")

            agent = GPILSContinuousAction(
            env=env,
            gamma=0.99,
            learning_rate=3e-4,
            learning_starts=1000,
            gradient_updates=10,
            policy_noise=0.2,
            net_arch=[256, 256, 256],
            project_name="ShowerRL",
            experiment_name=f"gpi_ls_model3_configB",
            use_gpi=False,
        )
            
        else:
            model_path = os.path.join(selector_path, f"gpi_ls_model3_configB.zip")
         
            agent = GPILS(
            env=env,
            gamma=0.99,
            learning_rate=3e-4,
            learning_starts=1000,
            gradient_updates=10,
            net_arch=[256, 256, 256],
            project_name="ShowerRL",
            experiment_name=f"gpi_ls_model3_configB",
            use_gpi=False,
        )
            
        

        
        print(f"Carregando modelo GPILSContinuousAction de: {model_path}")
        agent.load(model_path + f"/gpi_ls_model3_configB.tar")
        print("Modelo GPILSContinuousAction carregado com sucesso!")
            

    except Exception as e:
        print(f"ERRO: Falha ao restaurar o checkpoint de '{model_path}'.")
        print(f"Detalhes do erro: {e}")
        print("Verifique o conteúdo do diretório para confirmar se os arquivos de checkpoint estão presentes.")
        return

    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em '{model_path}'")
        return
    
    
    print("Agente restaurado com sucesso!")

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
            # Para GPIPDContinuousAction, forneça um vetor de pesos `w` para o predict
            # Ex: [0.8, 0.2] -> 80% de importância para temp, 20% para vazão
            w = weights_avaliacao 
            # w = np.array([1])  #### Remover depois de testar o IQB ####
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
        #  If utilizado apenas para colapsar todo o código
        if True:
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

            weights_str = ''.join(map(str, weights_avaliacao))

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
            ax[2].plot(tempo_acoes, custo_total_list_acumulado, label="Custo total", color="black", linestyle="dashed")
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
    parser.add_argument("nome_algoritmo", help="Nome do algoritmo", choices=("ppo", "sac", "gpils"))
    parser.add_argument("treina", help="Treina o agente", choices=("True", "False"))
    parser.add_argument("avalia", help="Avalia o agente", choices=("True", "False"))
    parser.add_argument("selector", help="Avalia o agente", choices=("True", "False"))
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
        # model = [banho_dia_frio]

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
        weights = [[1,0],
                    [0.8,0.2],
                    [0.6,0.4],
                    [0.5,0.5],
                    [0.4,0.6],
                    [0.2,0.8],
                    [0,1]]
        # weights = [[1,0]]
        
        for i in range(len(weights)):
            for j, k in combs:
                Tinf_val = float(j)
                custo_eletrico_kwh_val = float(k)
                resultados_list, concepts_list = avalia_agente(nome_algoritmo, [Tinf_val], [custo_eletrico_kwh_val], selector, weights[i])
                df_resultados.loc[len(df_resultados)] = resultados_list + [None] * (len(df_resultados.columns) - len(resultados_list))
                df_concepts.loc[len(df_concepts)] = concepts_list + [None] * (len(df_concepts.columns) - len(concepts_list))


        # Salva os resultados principais em um arquivo csv:
        if selector:
            df_resultados.to_csv("./resultados_tabela_selector/resultados_tabela_Tinf" + str(Tinf_val) + ".csv", index=False)
            df_concepts.to_csv("./resultados_concepts_selector/resultados_concepts_Tinf" + str(Tinf_val) + ".csv", index=False)
        else:
            df_resultados.to_csv("./resultados_tabela_programmed/resultados_tabela.csv", index=False)
