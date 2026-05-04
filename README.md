## Instructions to train the agent

## Setup and run

Considering that your environment is Visual Studio Code with Python, follow [this instructions](https://github.com/ray-project/ray/tree/master/rllib#installation-and-setup) to install the library in an Anaconda environment:

```bash
conda create -n rllib python=3.8
conda activate rllib
```

Enter the directory where your repository was cloned:

```bash
cd C:\directory\of\your\repo
```

Run the prompt below, choosing between True or False in the fields <train> and <eval>

```bash
python treina_agente_v3.py gpils <train> <eval>
```
