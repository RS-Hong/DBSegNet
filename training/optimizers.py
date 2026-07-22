"""Optimizer construction and batch-size-aware learning rates."""

import torch.optim as optim


def build_optimizer(model, config, learning_rate):
    choices = {
        "adam": lambda: optim.Adam(
            model.parameters(),
            learning_rate,
            betas=(config.momentum, 0.999),
            weight_decay=config.weight_decay,
        ),
        "adamw": lambda: optim.AdamW(
            model.parameters(),
            learning_rate,
            betas=(config.momentum, 0.999),
            weight_decay=config.weight_decay,
        ),
        "sgd": lambda: optim.SGD(
            model.parameters(),
            learning_rate,
            momentum=config.momentum,
            nesterov=True,
            weight_decay=config.weight_decay,
        ),
    }
    return choices[config.optimizer]()


def fitted_learning_rates(config):
    nominal_batch_size = 16
    maximum = 1e-4 if config.optimizer in {"adam", "adamw"} else 5e-2
    minimum = 3e-5 if config.optimizer in {"adam", "adamw"} else 5e-4
    initial = min(
        max(config.batch_size / nominal_batch_size * config.learning_rate, minimum),
        maximum,
    )
    final = min(
        max(
            config.batch_size / nominal_batch_size * config.min_learning_rate,
            minimum * 0.01,
        ),
        maximum * 0.01,
    )
    return initial, final
