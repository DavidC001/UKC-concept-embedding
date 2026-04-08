"""Main entry point for standalone WSD training."""

import WSD.config as config
from WSD.experiments import run_experiments
from WSD.training import train
from WSD.evaluation import evaluate

def _cmd_train(cfg: config.WSDConfig):
    train(cfg)


def _cmd_experiments(cfg: config.WSDConfig):
    run_experiments(limit=cfg.EXPERIMENTS_LIMIT)


def _cmd_quick(cfg: config.WSDConfig):
    quick_cfg = config.cfg_replace(
        cfg,
        NUM_EPOCHS=1,
        BATCH_SIZE=min(cfg.BATCH_SIZE, 8),
        EVAL_BATCH_SIZE=min(cfg.EVAL_BATCH_SIZE, 8),
    )
    _cmd_train(quick_cfg)

def _cmd_eval(cfg: config.WSDConfig):
    evaluate(cfg)

def main():
    cfg = config.parse_cli_config()
    
    if cfg.MODE == "train":
        _cmd_train(cfg)
    elif cfg.MODE == "experiments":
        _cmd_experiments(cfg)
    elif cfg.MODE == "quick":
        _cmd_quick(cfg)
    elif cfg.MODE == "eval":
        _cmd_eval(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.MODE}")


if __name__ == "__main__":
    main()
