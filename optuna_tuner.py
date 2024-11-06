import os

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig

import train


def sample_dirichlet(trial, n, name):

    a = []
    for i in range(n):
        a.append(- np.log(trial.suggest_float(f"{name}_unnormalized_{i}", 0, 1)))

    b = []
    for i in range(n):
        b.append(a[i] / sum(a))

    for i in range(n):
        trial.set_user_attr(f"{name}_{i}", b[i])

    return b


def lr_weight_decay_objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:

    modified_cfg = cfg.copy()

    # We optimize the number of layers, hidden units in each layer and dropouts.
    eff_bs = cfg.dataloader.effective_batch_size
    upper_lr = 1e-3 / (eff_bs/2)  # Found out due to stability issues.
    lower_lr = 1e-5 / (eff_bs/2)
    modified_cfg.optimization.lr = trial.suggest_loguniform("lr", lower_lr, upper_lr)
    modified_cfg.optimization.weight_decay = trial.suggest_loguniform("weight_decay", 0.0002, 0.2)
    # modified_cfg.optimization.patience = trial.suggest_categorical("patience", choices=[0, 1, 3])
    metric = train.train(modified_cfg, trial=trial)

    return metric


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    DATABASE_OPTIONS = {
        # DEMO
        # "MYSQL": "mysql://USER@localhost/optuna",
        # "SQLITE": "sqlite://///home/USER/optuna.db",
        "SYS_DEFAULT": os.environ["OPTUNA_D_BASE_LOC"],
    }

    cfg_optuna = cfg.optuna
    cfg_optuna.storage = DATABASE_OPTIONS.get(cfg_optuna.storage, cfg_optuna.storage)
    pruner = hydra.utils.instantiate(cfg_optuna.pruner)

    def _print_progress_callback(study: optuna.study, trial):
        print(f'Progress of study "{cfg_optuna.study_name}" in database {cfg_optuna.storage}')

        print(f"STARTED {len(study.trials)}/{cfg_optuna.num_trials}")
        print("---------------------------------------------")

        # TrialState is enum
        state_count = {enu.name: 0 for enu in optuna.trial.TrialState}

        for trial in study.trials:
            state_count[trial.state.name] += 1

        for state in state_count:
            print(state + " " + str(state_count[state]))

        print("---------------------------------------------")

        try:
            print(f"Best params: {study.best_params} with value {study.best_value} in trial")
            print(study.best_trial)
        except:
            print("No trial completed yet!")

    if cfg_optuna.delete_study:
        optuna.delete_study(cfg_optuna.study_name, cfg_optuna.storage)
        print(f"Deleted study: {cfg_optuna.study_name} in storage: {cfg_optuna.storage}")

        exit()

    if cfg_optuna.print_progress:

        study = optuna.load_study(
            study_name=cfg_optuna.study_name,
            storage=cfg_optuna.storage,
        )

        _print_progress_callback(study, None)
        exit()

    study = optuna.create_study(
        storage=cfg_optuna.storage,
        pruner=pruner,
        study_name=cfg_optuna.study_name,
        direction=cfg_optuna.direction,
        load_if_exists=True,
    )

    obj_dict = {
        "lr_weight_decay": lr_weight_decay_objective
    }

    objective = obj_dict[cfg_optuna.objective_function]

    def opti_function(trial):
        return objective(trial, cfg=cfg, optuna_target=cfg_optuna.target)

    study.optimize(opti_function, cfg_optuna.num_trials, catch=(ValueError,), callbacks=[_print_progress_callback])


if __name__ == "__main__":
    main()
