import hydra
from omegaconf import DictConfig

from train import train


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


main()
