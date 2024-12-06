# UnityMARL
Package a Unity environment into Python and implement multi-agent reinforcement learning

## Content
**combat.app** is a custom Unity environment that runs on MacOS on Apple Sillicon.<br>
**Combat_win** is a custom Unity environment that runs on Windows.<br>
In **uniy_wrapper** shows how to use mlagents-env to wrap Unity programs into Python code.<br>
**xuance_toturial** shows how to run the xuance deep reinforcement learning algorithm package and run the algorithm in a customized environment.<br>

## Installation
**Step 1**: Create a new conda environment (python=3.10 is suggested):
```commandline
conda create -n UnityMARL python==3.10.12
```
**Step 2**: Activate conda environment:
```commandline
conda activate UnityMARL
```
**Step 3**: Install pytorch:
```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
**Step 4**: Install xuance:
```commandline
cd UnityMARL/libs/xuance
pip install -e .
```
**Step 5**: Install mlagents-envs:
```commandline
pip install mlagents-envs
```

## Run
```commandline
cd examples/QMIX
python unity_env_QMIX.py
```
