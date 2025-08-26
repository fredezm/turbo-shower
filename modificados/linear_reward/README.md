# Modelos de RL treinados com os algoritmos PPO e SAC

## Modelos

### Primeira fase do estudo - otimização de qualidade

- [Configuração A da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_supervisoria_iqb)
- [Configuração B da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_rto_iqb_com_split_range)
- [Configuração C da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_rto_iqb_sem_split_range)

### Segunda fase do estudo - otimização de qualidade e custos

- [Modelo 1 da configuração B da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_rto_iqb_com_split_range_multiple_scenarios)
- [Modelo 2 da configuração B da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_rto_com_split_range_multiple_scenarios)
- [Modelos 3 e 4 da configuração B da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/multi_agent_camada_rto_iqb_com_split_range)
- [Modelos 5 e 6 da configuração B da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/multi_agent_camada_rto_com_split_range)
- [Modelo 1 da configuração C da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_rto_iqb_sem_split_range_multiple_scenarios)
- [Modelo 2 da configuração C da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/single_concept_camada_rto_sem_split_range_multiple_scenarios)
- [Modelos 3 e 4 da configuração C da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/multi_agent_camada_rto_iqb_sem_split_range)
- [Modelos 5 e 6 da configuração C da planta de chuveiro](https://github.com/mpaulazamin/tcc-final-models/tree/multi_agent_camada_rto_sem_split_range)

## Instalação

Considerando que o seu ambiente é o Visual Studio Code com Python, siga [estas instruções](https://github.com/ray-project/ray/tree/master/rllib#installation-and-setup) para instalar a biblioteca em um ambiente Anaconda:

```bash
conda create -n rllib python=3.8
conda activate rllib
pip install "ray[rllib]" tensorflow 
```

Depois, instale o pacote `tensorflow_probability` no mesmo ambiente, referenciado [neste link](https://github.com/tensorflow/probability/releases):

```bash
pip install tensorflow-probability==0.19.0
```

## Integração com Tensorboard

Siga [estas instruções](https://stackoverflow.com/questions/45095820/tensorboard-command-not-found) para integrar seus modelos com o Tensorboard:

Execute o seguinte comando:

```bash
pip show tensorflow
```

Entre no local onde o `tensorflow` está instalado:

```bash
cd C:\users\maria\appdata\roaming\python\python38\site-packages
```

Entre no folder do `tensorboard`:

```bash
cd tensorboard
```

Execute o seguinte comando:

```bash
python main.py --logdir "C:\users\maria\ray_results\folder_experiment"
```

## Sanity check

Treinando o sistema com o ambiente customizado [neste script](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/sanity_check.py), obtém-se o resultado abaixo, 
que reproduz o resultado encontrado no [notebook](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/chuveiro_turbinado.ipynb) corrigido pelo professor.

![check](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/imagens/custom_env.jpg)
