from comet_ml import Experiment
from data.data_utils import get_transforms, get_classes, get_open_set_classes
from data.dataloader import get_dataloaders
from modules.prompts import (
    visual_prompt,
    text_prompt,
)
from modules.clip_modules import ImageCLIP, TextCLIP
from modules.kll_loss import KLLoss
from utils.solver import _optimizer, _lr_scheduler
from utils.utils import (
    epoch_saving,
    best_saving,
    process_run_name,
)
from utils.attributes import k_means
from loops.test_step import test_step
from omegaconf import DictConfig, OmegaConf, open_dict
from os.path import join
from pathlib import Path
from shutil import copy
from termcolor import colored

import warnings
import wandb
import torch
import clip
import hydra
import sys


def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):

    warnings.simplefilter("ignore", ResourceWarning)

    seed_everything(config.seed)

    from loops.training_step import training_step

    # logger
    run_name, run_id, run_name_no_id = process_run_name(config)
    print(colored("RUN ID: {}".format(run_id), "green"))
    print(colored("RUN NAME: {}".format(run_name), "green"))
    if config.logging.wandb:
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=dict(config),
        )
    run = None

    experiment = None
    if config.logging.comet:
        experiment = Experiment(
            api_key="wIf8mPYXi87PpERUtIBhKOX3c",
            project_name=config.logging.project_name,
            workspace="gzaraunitn",
        )
        experiment.set_name(name=run_name)
        experiment.log_parameters(dict(config))
        experiment.log_parameter("command", " ".join(sys.argv))
        if config.logging.tag:
            experiment.add_tag(config.logging.tag)

    # config file
    working_dir = join(
        "./exp",
        run_name_no_id,
        run_id,
    )

    # generic logging
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(config):
        config.log_file_path = join(working_dir, "log.txt")
    copy("configs/config.yaml", working_dir)
    copy("train.py", working_dir)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # clip model
    model, clip_state_dict = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.dropout,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint=config.network.joint,
    )  # Must set jit=False for training  ViT-B/32

    # fetch training and test transformations
    training_transforms, test_transforms = get_transforms(config)

    # clip modules
    video_model = ImageCLIP(model)
    fusion_model = visual_prompt(
        config.network.sim_header, clip_state_dict, config.data.num_segments
    )
    text_model = TextCLIP(model)

    # wandb watch models
    if config.logging.wandb:
        wandb.watch(model)
        wandb.watch(fusion_model)

    # fetch dataloaders
    # train_loader, val_loader, source_loader, kmeans_loader = get_dataloaders(
    dataloaders = get_dataloaders(
        config=config,
        training_transforms=training_transforms,
        test_transforms=test_transforms,
    )

    # weights type conversions
    if device == "cpu":
        text_model.float()
        video_model.float()
    else:
        clip.model.convert_weights(
            text_model
        )  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(video_model)

    # define loss
    loss_video = KLLoss()
    loss_text = KLLoss()

    # starting epoch and best precision
    start_epoch = config.solver.start_epoch
    best_score = 0.0

    # text prompt
    classes_names = get_classes(config)
    prompts = text_prompt(classes_names)
    prompts["classes_names"] = classes_names
    if config.open_set.method == "oracle":
        print(colored("Fetching open-set class names...", "green"))
        prompts["open_set_labels"] = get_open_set_classes(config)
        print(colored("Open-set class names fetched", "green"))

    # load pretrained model
    if config.network.pretrained_model:
        print(colored("Loading pretrained weights...", "green"))
        full_model_params = torch.load(
            config.network.pretrained_model, map_location="cpu"
        )["model_state_dict"]
        model.load_state_dict(full_model_params)

    # clustering method
    if config.attributes.clustering_method == "kmeans":
        clustering_method = k_means
    else:
        raise ValueError("Clustering method {} not recognized!".format(config.attributes.clustering_method))

    # group models
    models = {
        "full": model,
        "video_model": video_model,
        "text_model": text_model,
        "fusion_model": fusion_model
    }

    for m in models:
        models[m] = torch.nn.DataParallel(models[m]).cuda()

    if config.network.pretrained_model:
        print(colored("Loading fusion model pretrained weights...", "green"))
        fusion_model_params = torch.load(
            config.network.pretrained_model, map_location="cpu"
        )["fusion_model_state_dict"]
        models["fusion_model"].load_state_dict(fusion_model_params)
        print(colored("Weights loaded", "green"))

    # group losses
    loss = {"loss_video": loss_video, "loss_text": loss_text}

    # optimizer and scheduler
    optimizers = _optimizer(config, models)
    scheduler = None
    if config.solver.type != "monitor":
        scheduler = _lr_scheduler(config, optimizers["main_optimizer"])

    samples_per_pseudo_labels = None

    for epoch in range(start_epoch, config.solver.epochs):

        if epoch == 0 and config.general.sanity_check:
            _ = test_step(
                models,
                dataloaders,
                config,
                prompts,
                device,
                classes_names,
                epoch,
                run_id,
                sanity_check=True,
            )
            print(colored("Sanity check ok!\n", "green"))

        training_results = training_step(
            models,
            dataloaders,
            clustering_method,
            optimizers,
            scheduler,
            config,
            prompts,
            device,
            loss,
            epoch,
            run,
            experiment,
            run_id,
            samples_per_pseudo_labels,
        )

        samples_per_pseudo_labels = training_results["samples_per_pseudo_labels"]

        score = test_step(
            models,
            dataloaders,
            config,
            prompts,
            device,
            classes_names,
            epoch,
            run_id,
            training_results,
            run=run,
            experiment=experiment,
        )

        is_best = score > best_score
        best_score = max(score, best_score)

        line = "Current/Best: {}/{}\n".format(score, best_score)
        print(line)
        with open(config.log_file_path, "a") as logfile:
            logfile.write("{}\n\n".format(line))

        # log best score
        if config.logging.comet:
            experiment.log_metric("best_score", best_score, step=epoch)

        if config.logging.save:
            print("Saving...")
            filename = "{}/last_model.pt".format(working_dir)

            epoch_saving(
                epoch, model, fusion_model, optimizers["main_optimizer"], filename
            )
            if is_best:
                best_saving(
                    working_dir,
                    epoch,
                    model,
                    fusion_model,
                    optimizers["main_optimizer"],
                )
            print("Saved\n")

    if config.logging.comet:
        experiment.end()


if __name__ == "__main__":
    main()
