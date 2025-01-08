import hydra
from trainer import Trainer
from termcolor import colored

@hydra.main(version_base="1.3", config_path="conf", config_name="base")
def main(cfg):

    config_name = cfg.get("mode", "train")
    if "train" in config_name:
        print(colored("In training mode.", "yellow"))
        trainer = Trainer(cfg)
        trainer.train_loop()

    elif "test" in config_name:
        print(colored("In testing mode.", "yellow"))
        ckpt = cfg.get("ckpt", None)
        if ckpt is not None:
            trainer = Trainer.load(cfg.ckpt, cfg)
        else:
            raise Exception("No checkpoint found for testing")
    else:
        raise Exception("Invalid configuration file name")


if __name__ == "__main__":
    main()
