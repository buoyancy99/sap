# Scoring-Aggregating-Planning
This is the master repo for Zero-shot Policy Learning with Spatial Temporal RewardDecomposition on Contingency-aware Observation
[[arxiv]](https://arxiv.org/abs/1910.08143)

# Setup
```
CLONE THIS REPO;
cd INTO THIS REPO
export PYTHONPATH=$PYTHONPATH:.

conda create -n SAP python=3.6
conda activate SAP
conda install cudatoolkit=10.0
conda install cudnn=7.6.0
pip3 install -r requirements.txt
```

# Reproduce our result
## Grid World
### Gridworld with SAP
```
python3 mpc/gridworld/mpc.py
```

### Gridworld with Imitation
```
python3 imitation/gridworld/imitation_model.py
```

## Mario
### Option: --world 1 (train) or --world 2 (test)
### Mario SAP
```
python3 mpc/mario/mpc_dynamics.py --world [1 or 2]
```

### Mario Rudder
```
python3 mpc/mario/rudder_mpc_dynamics.py --world [1 or 2]
```

### Mario DARLA
```
python3 imitation/mario/bench_imitation_darla.py --world [1 or 2]
```

### Mario Behavior Cloning (BC)
```
python3 imitation/mario/bench_imitation.py --world [1 or 2]
```

### Mario MBHP
```
python3 mpc/mario/mpc_dynamics.py --mbhp --world [1 or 2]
```

### Mario Privileged Behavior Cloning (Priv_BC)
```
python3 imitation/mario/bench_imitation_expert.py --world [1 or 2]
```

### Ablation: Mario SAP without spatial reward decomposition (SAP w/o spatial)
```
python3 mpc/mario/nogrid_mpc_dynamics.py --world [1 or 2]
```

### Ablation: Mario SAP without spatial nor temporal reward decomposition (SAP w/o spatial temporal)
```
python3 mpc/mario/nogrid_mpc_dynamics.py --mbhp --world [1 or 2]
```

### Ablation: GT dynamics model, Mario SAP
```
python3 mpc/mario/mpc_perfect.py --world [1 or 2]
```

### Ablation: GT dynamics model, Mario Rudder
```
python3 mpc/mario/rudder_mpc_perfect.py --world [1 or 2]
```

### Ablation: GT dynamics model, Mario MBHP
```
python3 mpc/mario/mpc_perfect.py --mbhp --world [1 or 2]
```

### Ablation: GT dynamics model & no done signal, Mario SAP
```
python3 mpc/mario/mpc_perfect.py --nodeath --world [1 or 2]
```

### Ablation: GT dynamics model & no done signal, Mario Rudder
```
python3 mpc/mario/rudder_mpc_perfect.py --nodeath --world [1 or 2]
```

### Ablation: GT dynamics model & no done signal, Mario MBHP
```
python3 mpc/mario/mpc_perfect.py --mbhp --nodeath --world [1 or 2]
```

### Ablation: Effect of planning steps:
```
# simply add option --plan_step to command
# for example
python3 mpc/mario/mpc_dynamics.py --plan_step 8 --world [1 or 2]
```


# Steps for re-training a policy
To re-train a policy, you generally need to: Generate imperfect trajectory rollouts; Preprocess rollouts into tfrecord for differnt kinds of experiments; Train reward decomposition and dynamics model for MPC or train imitation for BC based methods; Benchmark learned policy to get results. In general, generate imperfect rollouts take a long time and the file is too big to upload here and we may archive them on public server upon acceptance. However all code for these steps are contained in this repo and you can generate rollouts yourself with enough computing power.
We assume the dataset to be put at a fixed position, specified by an env variable ''DATASET_ROOT''

## Mario
random-network-distillation folder folder contains code you need for mario to train a policy and generate imperfect rollouts. You may refer to README inside or the original repo for instructions.
Please put generated rollouts in ```DATASET_ROOT/Datasets/MarioRaw```

```
# set dataset root env variable
export DATASET_ROOT=/home/[yourusername]

# generate tf records for reward learning
python3 prior_learning/mario/preprocess.py

# train reward learning for differnt experiments
python3 prior_learning/mario/train.py
python3 prior_learning/mario/rudder_train.py
python3 prior_learning/mario/nogrid_train.py

# generate tf records for dynamics model learning
python3 dynamics/mario/preprocess.py

# train dynamics model for mario
python3 dynamics/mario/dynamics_model.py

# generate tf records for bc based policy
python3 imitation/mario/preprocess.py
python3 imitation/mario/preprocess_darla.py
python3 imitation/mario/preprocess_expert.py
python3 imitation/mario/preprocess_expert_local.py
python3 imitation/mario/preprocess_local.py

# train bc based policy
python3 imitation/mario/imitation_model.py
python3 imitation/mario/imitation_model_darla.py
python3 imitation/mario/imitation_model_expert.py
python3 imitation/mario/imitation_model_expert_local.py
python3 imitation/mario/imitation_model_local.py

# train DARLA
python3 DARLA/mario/train.py

``` 
And then you will have all the models you need for "reproduce our result" section.





