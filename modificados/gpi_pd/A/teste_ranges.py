import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

from treina_agente import ShowerEnv

def find_range(num_episodes=100):
    """
    Runs a fwe episodes with random actions to find the range of IQB values.
    """
    env = gym.make('Shower-v0', env_config={"Tinf": 30, "nome_algoritmo": "global_policy_improvement"})
    iqb_values= []
    ts_values = []
    fs_values = []

    for i in range(num_episodes):
        obs, info = env.reset()
        terminated = False 
        truncated = False
        while not terminated and not truncated:
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # Collect the IQB value
            iqb_values.append(info["iqb"])
            ts_values.append(info["Ts"])
            fs_values.append(info["Fs"])
    env.close()

    min_iqb = np.min(iqb_values)
    max_iqb = np.max(iqb_values)

    min_fs = np.min(fs_values)
    max_fs = np.max(fs_values)

    min_ts = np.min(ts_values)
    max_ts = np.max(ts_values)

    print(f"IQB mínimo observado: {min_iqb}")
    print(f"IQB máximo observado: {max_iqb}")

    print(f"Fs mínimo observado: {min_fs}")
    print(f"Fs máximo observado: {max_fs}")

    print(f"Ts mínimo observado: {min_ts}")
    print(f"Ts máximo observado: {max_ts}")
    return min_iqb, max_iqb

if __name__ == '__main__':
    min_iqb, max_iqb = find_range(num_episodes=10000)