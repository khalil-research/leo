# LEO: Learning Efficient Orderings for Multiobjective Binary Decision Diagrams
*Official implementation* 

Full paper: https://arxiv.org/abs/2307.03171

**Abstract**: Approaches based on Binary decision diagrams (BDDs) have recently achieved state-of-the-art results for multiobjective integer programming problems. The variable ordering used in constructing BDDs can have a significant impact on their size and on the quality of bounds derived from relaxed/restricted BDDs for single-objective optimization problems. We first showcase a similar impact for variable ordering on the Pareto frontier enumeration time for the multiobjective knapsack problem, suggesting the need for deriving VO methods that improve the scalability of the multiobjective BDD approach. To that end, we derive a novel parameter configuration space based on variable scoring functions which are linear in a small set of interpretable and easy-to-compute variable features. We show how the configuration space can be efficiently explored using black-box optimization, circumventing the curse of dimensionality (in the number of variables/objectives), and finding good orderings that reduce the Pareto frontier enumeration time. However, black-box optimization approaches incur a computational overhead that outweighs the reduction in time due to good variable ordering. To alleviate this issue, we propose `LEO`, a supervised learning approach for finding efficient variable orderings that reduce the enumeration time. Results on benchmark sets from the knapsack problem with 3-7 objectives and up to 80 variables show that the learning-based method can significantly reduce the Pareto frontier enumeration time compared to common ordering strategies and algorithm configuration.

## Repository structure

LEO is structured as follows:

```

|-- code
    |-- cpp
    |-- python
        |-- leo
|-- resources
    |-- instances
    |-- bin
```

The `code/cpp` folder contains the code for BDD Manager, which is based on the implementation of [1] available at https://www.andrew.cmu.edu/user/vanhoeve/mdd/. Once the binary is successfully built, run the code in `code/python/leo` package for executing different phases of LEO. 

## Building the C++ code

Building the C++ code is straightforward, thanks to Andre and David. Go to `code/cpp` and correctly set the following items based on your system in `makefile`.

```
SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
BASISDIR   = /opt
BASISILOG  = $(BASISDIR)/ibm/ILOG/CPLEX_Studio1210
BOOSTDIR   = $(BASISDIR)/boost
```

Once everything is set execute `make`. If it ran correctly then there should be a binary named `multiobj` created in this folder. Copy this binary to `resources/bin`.

## Running the Python code

Hence forth we will assume that you are `code/python` folder. All the scripts named `leo/<filename>.py` have a corresponding config file in `leo/config/<filename>.yaml`. 

To get a list of all the commands to run, execute

```
python -m leo.get_cmd
```
You can alter `leo/config/get_cmd.yaml` to get commands for a subset of components of the entire pipeline. To further modify each component, for example setting the seeds on which you want to run SMAC or changing the parameters of the grid-search for training models, we strongly suggest taking a closer look at `leo/get_cmd.py` and modifying parameters as per your need. Running this command will output a file named `cmds.txt`. All the commands inside this file can be run sequentially by calling

```
python leo/runner.py -t <path-to-cmds.txt>
```

We now describe how to run each phase or any of its components individually.
### Phase 1: Data Labeling
This phase comprises of generating labels using SMAC. 

**1. Label an instance using SMAC, i.e., run SMAC in SmacI mode.**
```
python -m leo.label_instance mode=SmacI split=train from_pid=0 num_instances=1 seed=777 cutoff_time=60 wallclock_limit=300 mem_limit=16 problem.n_objs=3 problem.n_vars=60
```
Suppose `<size> = <n_objs>_<n_vars>`, then the SMAC output is stored in `resources/SmacI_out/<problem_name>/<size>/<split>/kp_7_<size>_<pid>/run_<seed>`.

Going forward, we will assume that `<problem_name>=knapsack` and `<size>=<problem.n_objs>_<problem.n_vars>`.

**2. Obtain SMAC baseline label, i.e., run SMAC in SmacD mode.**
```
python -m leo.label_instance mode=SmacD split=train from_pid=0 num_instances=1000 seed=777 cutoff_time=60 wallclock_limit=300 mem_limit=16 problem.n_objs=3 problem.n_vars=60
```
The SMAC output is stored in `resources/SmacD_out/<problem_name>/<size>/<split>/kp_7_<size>_<from_pid>/run_<seed>`.

### Phase 2: Dataset Generation and Model Training

**3. Find best label among all SmacI runs with different seeds.**

```
python -m leo.find_best_label
```

The output will be saved in `resources/labels/<problem_name>/<size>` directory.

**4. Generate dataset based based on labels.**

- Generate size-specific dataset, without context features.
```
python -m leo.generate_dataset task=pair_rank fused=0 context=0
python -m leo.generate_dataset task=point_regress fused=0 context=0
```

- Generate size-specific dataset, with context features.
```
python -m leo.generate_dataset task=pair_rank fused=0 context=1
python -m leo.generate_dataset task=point_regress fused=0 context=1
```

- Generate single dataset using all sizes, without context features.
```
python -m leo.generate_dataset task=pair_rank fused=1 context=0
```

- Generate single dataset using all sizes, with context features.
```
python -m leo.generate_dataset task=pair_rank fused=1 context=1
```

Each dataset is assigned a name called `<dataset_name>` based on `context` and `fused` parameters. For size-specific models, the `<dataset_name>` will be `<size>` (or `<size>_context`) if context features are not used (or used). Similarly, models trained on entire dataset will be named `all` (or `all_context`). 

The datasets will be stored in `resources/datasets/<problem_name>` folder. The datasets for `task=pair_rank` comprises of three files named as `<dataset_name>_<filetype>_pair_rank_<split>.dat`. Here `<filetype>` can be `dataset`, `names` or `n_items`. The datasets for `task=point_regress` are named as `<dataset_name>_dataset_point_regress_<split>.pkl`.

**5. Train learning models with different datasets and hyperparameters.**

```
python -m leo.train task=<task> model.name=<model_name> context=<context> fused=<fused> problem.n_objs=<n_objs> problem.n_vars=<n_vars> model.<hyperparam_1>=<hyperparam-val-1> ... <hyperparam_n>=<hyperparam-val-n>
```

Here `<task>` can be `point_regress` or `pair_rank` for training pointwise or pairwise ranking model, respectively. For `task=point_regress`, `<model_name>` can be `LinearRegression`, `Lasso`,`Ridge`, `DecisionTreeRegressor`,  or `GradientBoostingRegressor`. For `task=pair_rank`, `<model_name>` can be 
`SVMRank`, or `GradientBoostingRanker`. To set the hyperparameters of each model type you can set the flag `model.<hyperparam_*>`. Refer to `code/python/leo/config/model/<model_name>` to find the list of hyperparameters for each model type. 

The `<context>` flag set to `1` (or `0`) indicates that we will use a dataset with (or without) context features to train the model. 

The `<fused>` flag set to `1` (or `0`) indicates that we will use a dataset comprising of all sizes (or a specific size). When `fused=0`, the size of the dataset is detemined using the `problem.n_objs` and `problem.n_vars`.

Each model is identified using a unique hash called `model_id`, which based on `model-name`, `model.<hyperparam_1>`, ..., `model.<hyperparam_n>`. The model configs will be saved in `resources/model_cfg/<model_id>.yaml`. The predictions of the model on validation set and training metrics will be saved in `resources/predictions/<problem_name>/<dataset_name>/prediction_<model_id>.pkl` and `resources/predictions/<problem_name>/<dataset_name>/results_<model_id>.pkl`. 


### Phase 3: Model Selection and Testing

**6. Find best model based on Kendall's Tau [2] ranking metric.**
```
python -m leo.find_best_model
```
The outputs will be saved in `resources/model_summary/<problem_name>`.

The `summary.csv` contains details about all the models trained across different task, dataset and model types.

The `<dataset_name>.csv` contains details about all the models trained for a given dataset type. 

The `best_model_<dataset_name>.csv` contains details about the best model for each model type for a given dataset type.


**7. Use the best model to make predictions on the test split.**

```
python -m leo.test mode=<mode> model_id=<model_id> task=<task> fused=<fused> context=<context> problem.n_objs=<n_objs> problem.n_vars=<n_vars>
```

The `mode` can be set to `best`, `all` or `one`. When `mode=best`, all the models in `best_model_<dataset_name>.csv` trained for `task=<task>` will be used to make predictions on the test set. When `mode=all`, all the models in `<dataset_name>.csv` will be used to make predictions on the test set. When `mode=one`, only model with `model_id=<model_id>` will be used to make predictions. 

For saving variable orderings of different heuristics set `model_name` to `Random`, `Lex`, `HeuristicValue`, `HeuristicWeight`, or `HeuristicValueByWeight`. The parameters of the heuristic orderings can be found in `leo/config/<heuristic_name>.yaml`.

Running this command will update the `prediction_<model_id>.pkl` and `results_<model_id>.pkl` files with the output on the test set. 

**8. Evaluate predicted variable ordering.**
```
python -m leo.eval_order task=<task> mode=<mode> model_name=<model_name> model_id=<model_id> split=<split> from_pid=<from_pid> to_pid=<to_pid> 
```

The `mode` can be set to `best`, or `one`. If `mode=best`, then the predictions of the best `<model_name>` model trained on `<task>` task will be used building the binary decision diagrams. If `mode=one`, then the predictions of the model `<model_id>` trained on `<task>` task will be used.

The output will be saved in `resources/eval_order/<dataset_name>`.

## References

[1] Bergman, David, and Andre A. Cire. "Multiobjective optimization by decision diagrams." Principles and Practice of Constraint Programming: 22nd International Conference, CP 2016, Toulouse, France, September 5-9, 2016, Proceedings 22. Springer International Publishing, 2016.

[2] Kendall, Maurice G. "A new measure of rank correlation." Biometrika 30.1/2 (1938): 81-93.