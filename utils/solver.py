import torch.optim as optim
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def _optimizer(config, models):

    optimizers = {}

    if config.network.freeze == "none":
        params = [
            {"params": models["full"].parameters()},
            {
                "params": models["fusion_model"].parameters(),
                "lr": config.solver.lr * config.solver.f_ratio,
            },
        ]
    else:
        params = []
        to_freeze = config.network.freeze.split("+")
        for module in to_freeze:
            assert module in models, "Module {} not in models!".format(module)
            for p in models[module].parameters():
                p.requires_grad = False
        if "text_model" not in to_freeze:
            params.append({"params": models["text_model"].parameters()})
        if "video_model" not in to_freeze:
            params.append({"params": models["video_model"].parameters()})
        if "fusion_model" not in to_freeze:
            params.append(
                {
                    "params": models["fusion_model"].parameters(),
                    "lr": config.solver.lr * config.solver.f_ratio,
                }
            )

    if config.solver.optim == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.solver.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.2,
        )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print("Adam")
    elif config.solver.optim == "sgd":

        optimizer = optim.SGD(
            params,
            config.solver.lr,
            momentum=config.solver.momentum,
            weight_decay=config.solver.weight_decay,
        )
        print("SGD")
    elif config.solver.optim == "adamw":
        vision_params = list(map(id, models["full"].module.visual.parameters()))
        text_params = filter(
            lambda p: id(p) not in vision_params, models["full"].parameters()
        )
        to_freeze = config.network.freeze.split("+")
        params = []
        if "text_model" not in to_freeze:
            params.append({"params": text_params, "lr": config.solver.lr})
        if "video_model" not in to_freeze:
            params.append(
                {"params": models["full"].module.visual.parameters(), "lr": config.solver.lr}
            )
        if "fusion_model" not in to_freeze:
            params.append(
                {
                    "params": models["fusion_model"].parameters(),
                    "lr": config.solver.lr * config.solver.f_ratio,
                }
            )

        optimizer = optim.AdamW(
            params,
            betas=(0.9, 0.98),
            lr=config.solver.lr,
            eps=1e-8,
            weight_decay=config.solver.weight_decay,
        )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        # for param_group in optimizer.param_groups:
        #     print(param_group["lr"])
        # print("AdamW")
    else:
        raise ValueError("Unknown optimizer: {}".format(config.solver.optim))

    optimizers["main_optimizer"] = optimizer

    return optimizers


def _lr_scheduler(config, optimizer):
    if config.solver.type == "cosine":
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer, config.solver.epochs, warmup_epochs=config.solver.lr_warmup_steps
        )
    elif config.solver.type == "multistep":
        lr_decay_steps = list(config.solver.lr_decay_steps)
        if len(lr_decay_steps) == 1:
            lr_decay_steps = int(lr_decay_steps[0])
        if isinstance(lr_decay_steps, list):
            milestones = lr_decay_steps
        elif isinstance(lr_decay_steps, int):
            milestones = [
                lr_decay_steps * (i + 1)
                for i in range(config.solver.epochs // lr_decay_steps)
            ]
        else:
            raise ValueError(
                "error learning rate decay step: {}".format(type(lr_decay_steps))
            )
        lr_scheduler = WarmupMultiStepLR(
            optimizer, milestones, warmup_epochs=config.solver.lr_warmup_steps
        )
    else:
        raise ValueError("Unknown lr scheduler: {}".format(config.solver.type))
    return lr_scheduler
