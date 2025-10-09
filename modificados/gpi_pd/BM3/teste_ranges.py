import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

from treina_agente import ShowerEnv

def find_range(num_episodes=100):
    """
    Runs a fwe episodes with random actions to find the range of IQB values.
    """
    
    Tinf_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]
    env_config = {"env_config": {"Tinf_list": Tinf_list,
                                         "nome_algoritmo": "gpi-ls",
                                         "custo_eletrico_kwh_list": custo_eletrico_kwh_list},
                        "only_iqb": True,}
    env = gym.make("Shower-v0", **env_config)
    iqb_values= []
    xq_values = []
    xf_values = []
    ff_values = []
    fq_values = []

    for i in range(num_episodes):
        obs, info = env.reset()
        terminated = False 
        truncated = False
        while not terminated and not truncated:
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # Collect the IQB value
            # iqb_values.append(info["iqb"])
            ff_values.append(info["Ff"])
            fq_values.append(info["Fq"])
            xq_values.append(info["xq"])
            xf_values.append(info["xf"])
    env.close()

    # min_iqb = np.min(iqb_values)
    # max_iqb = np.max(iqb_values)

    min_xq = np.min(xq_values)
    max_xq = np.max(xq_values)

    min_xf = np.min(xf_values)
    max_xf = np.max(xf_values)
    
    min_fq = np.min(fq_values)
    max_fq = np.max(fq_values)

    min_ff = np.min(ff_values)
    max_ff = np.max(ff_values)

    # print(f"IQB mínimo observado: {min_iqb}")
    # print(f"IQB máximo observado: {max_iqb}")

    print(f"xq mínimo observado: {min_xq}")
    print(f"xq máximo observado: {max_xq}")

    print(f"xf mínimo observado: {min_xf}")
    print(f"xf máximo observado: {max_xf}")
    
    print(f"Fq mínimo observado: {min_fq}")
    print(f"Fq máximo observado: {max_fq}")

    print(f"Ff mínimo observado: {min_ff}")
    print(f"Ff máximo observado: {max_ff}")
if __name__ == '__main__':
    
    find_range(num_episodes=10000)