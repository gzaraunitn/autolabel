import torch
import numpy as np
import wandb
import clip
import warnings
from tqdm import tqdm
from collections import defaultdict
from json import dumps
from termcolor import colored
from utils.utils import (
    create_logits,
    gen_label,
    convert_models_to_fp32,
    AverageMeter,
    compute_accept_mask,
    compute_acc,
)
from utils.attributes import (
    compute_clustering_accuracy,
    extract_cluster_attributes,
    match_attributes,
    extract_class_attributes,
)


def training_step(
    model,
    loader,
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
    thresholds=None,
    samples_per_pseudo_labels=None,
):
    warnings.simplefilter("ignore", ResourceWarning)

    # define metrics to log
    train_total_loss_source = AverageMeter()
    train_loss_video_source = AverageMeter()
    train_loss_text_source = AverageMeter()
    train_total_loss_target = AverageMeter()
    train_loss_video_target = AverageMeter()
    train_loss_text_target = AverageMeter()
    train_total_loss_domain = AverageMeter()
    train_loss_video_domain = AverageMeter()
    train_loss_text_domain = AverageMeter()
    pseudo_label_acc = AverageMeter()
    entropy_loss_target = AverageMeter()
    consistency_loss_target = AverageMeter()
    accepted_pseudo_labels = 0
    total_pseudo_labels = 0

    # set models to train mode
    for m in model:
        model[m].train()

    # store confidence of pseudo labels
    new_samples_per_pseudo_labels = defaultdict(list)

    # clustering
    matching_results = None
    if config.open_set.method == "autolabel":
        clustering_results = clustering_method(model, loader["target"], config, device)
        (
            target_attributes_per_cluster,
            true_labels_per_cluster,
        ) = extract_cluster_attributes(clustering_results, config, epoch, run_id)

        # attributes extraction and matching
        print(colored("Exctracting attributes...", "green"))
        source_attributes_per_class = extract_class_attributes(config)
        print(colored("Attributes extracted. Matching attributes...", "green"))
        matching_results = match_attributes(
            source_attributes_per_class, target_attributes_per_cluster, config
        )
        clustering_accuracy = compute_clustering_accuracy(
            matching_results["matches_per_cluster_label"],
            clustering_results["labels"],
            clustering_results["true_labels"],
            config,
        )
        print("Clustering accuracy: {}".format(clustering_accuracy))
        if config.logging.wandb:
            wandb.log({"clustering_accuracy": clustering_accuracy})
        if config.logging.neptune:
            run["clustering_accuracy"].log(clustering_accuracy)
        if config.logging.comet:
            experiment.log_metric(
                "clustering_accuracy", clustering_accuracy, step=epoch
            )
        print(colored("Attributes matched", "green"))

        if config.logging.verbose:
            print(
                dumps(
                    matching_results["matches_per_cluster_label"],
                    indent=2,
                )
            )

    print("[Training] - Epoch {}".format(epoch + 1))
    for batch_idx, batch in enumerate(tqdm(loader["train"])):
        if "ek" in config.data.dataset:
            (
                video_source,
                y_source,
                id_source,
                video_target,
                y_target,
                _,
                _,
                _,
                id_target,
                target_index,
            ) = batch
        else:
            (
                video_source,
                y_source,
                id_source,
                video_target,
                y_target,
                _,
                id_target,
                target_index,
            ) = batch

        if config.solver.type != "monitor":
            if (batch_idx + 1) == 1 or (batch_idx + 1) % 10 == 0:
                warnings.simplefilter("ignore")
                scheduler.step(epoch + batch_idx / len(loader))
                warnings.resetwarnings()

        # zero the gradients
        optimizers["main_optimizer"].zero_grad()

        # --------------------------- SOURCE --------------------------- #

        # reshape the source input and get size
        video_source = video_source.view(
            (-1, config.data.num_segments, 3) + video_source.size()[-2:]
        )
        b, t, c, h, w = video_source.size()

        # build text data
        text_id = np.random.randint(prompts["num_text_aug"], size=len(y_source))
        texts_source = torch.stack(
            [prompts["text_dict"][j][i, :] for i, j in zip(y_source, text_id)]
        )

        # move to gpu device
        video_source = video_source.to(device).view(-1, c, h, w)
        texts_source = texts_source.to(device)

        # produce embeddings
        video_embedding = model["video_model"](video_source)
        video_embedding = video_embedding.view(b, t, -1)
        video_embedding = model["fusion_model"](video_embedding)
        text_embedding = model["text_model"](texts_source)

        # compute logit scale
        logit_scale = model["full"].module.logit_scale.exp()

        # create logits
        logits_per_video, logits_per_text = create_logits(
            video_embedding, text_embedding, logit_scale
        )

        # generate ground truth
        ground_truth = torch.tensor(
            gen_label(y_source), dtype=video_embedding.dtype, device=device
        )

        # compute loss
        loss_video_source = loss["loss_video"](logits_per_video, ground_truth)
        loss_text_source = loss["loss_text"](logits_per_text, ground_truth)
        total_loss_source = (loss_video_source + loss_text_source) / 2
        total_loss = config.loss.source.weight * total_loss_source

        # update counters
        assert logits_per_video.size(0) == logits_per_text.size(0)
        train_total_loss_source.update(
            total_loss_source.item(), logits_per_video.size(0)
        )
        train_loss_video_source.update(
            loss_video_source.item(), logits_per_video.size(0)
        )
        train_loss_text_source.update(loss_text_source.item(), logits_per_video.size(0))

        # --------------------------- TARGET --------------------------- #

        if config.loss.target.weight:

            # reshape the target input and get size
            video_target = video_target.view(
                (-1, config.data.num_segments, 3) + video_target.size()[-2:]
            )
            b, t, c, h, w = video_target.size()

            extended_classes = prompts["classes"]
            text_dict_target = prompts["text_dict_target"]

            if config.open_set.method in ["autolabel", "oracle"]:
                if config.open_set.method == "autolabel":
                    if config.attributes.clustering_method == "kmeans":
                        extended_classes_names = [
                            c for _, c in prompts["classes_names"]
                        ] + matching_results["open_set_labels"]
                    else:
                        raise ValueError(
                            "Clustering algorithm {} not recognized!".format(
                                config.attributes.clustering_method
                            )
                        )
                else:
                    extended_classes_names = [
                        c for _, c in prompts["classes_names"]
                    ] + prompts["open_set_labels"]
                text_dict_target = {}
                for i, txt in enumerate(prompts["text_aug"]):
                    text_dict_target[i] = torch.cat(
                        [clip.tokenize(txt.format(c)) for c in extended_classes_names]
                    )
                extended_classes = torch.cat([v for _, v in text_dict_target.items()])

                with open(
                    "{}_extended_classes.txt".format(run_id), "a"
                ) as extended_classes_file:
                    extended_classes_file.write(
                        "\n{}: {}\n".format(epoch, extended_classes_names)
                    )

            video_target = video_target.to(device).view(-1, c, h, w)
            video_embedding = model["video_model"](video_target)
            video_embedding = video_embedding.view(b, t, -1)
            video_embedding = model["fusion_model"](video_embedding)

            # compute pseudo labels and acceptance mask
            mask, accepted, pseudo_y_target, values = compute_accept_mask(
                model,
                video_embedding,
                extended_classes,
                prompts,
                device,
                config,
                thresholds,
                samples_per_pseudo_labels,
                target_index,
            )

            # gather target samples and their confidence
            for i in range(b):
                label = pseudo_y_target[i]
                entry = (target_index[i].item(), values[i].item())
                new_samples_per_pseudo_labels[label.item()].append(entry)

            # keep track of total pseudo labels
            total_pseudo_labels += b

            # build text data
            if accepted:

                # keep track of accepted pseudo labels
                accepted_pseudo_labels += accepted
                y_target = y_target.to(device)
                mask = mask.to(device)
                pseudo_y_target = pseudo_y_target.to(device)
                pseudo_label_accuracy = compute_acc(
                    y_target[mask].to(device), pseudo_y_target[mask]
                )
                pseudo_label_acc.update(
                    pseudo_label_accuracy, pseudo_y_target[mask].size(0)
                )

                # fetch target label
                if config.loss.target.use_gt:
                    target_label = y_target
                else:
                    target_label = pseudo_y_target

                text_id = np.random.randint(prompts["num_text_aug"], size=b)
                texts_target = torch.stack(
                    [text_dict_target[j][i, :] for i, j in zip(target_label, text_id)]
                )

                # move to gpu device
                mask = mask.to(device)
                texts_target = texts_target.to(device)
                texts_target = texts_target[mask].to(device)

                # produce embeddings
                video_embedding = video_embedding[mask]
                text_embedding = model["text_model"](texts_target)

                if config.loss.target.weight:

                    # create logits
                    logits_per_video, logits_per_text = create_logits(
                        video_embedding, text_embedding, logit_scale
                    )

                    # generate ground truth
                    if not config.loss.target.use_gt:
                        target_label = target_label[mask]
                    ground_truth = torch.tensor(
                        gen_label(target_label),
                        dtype=video_embedding.dtype,
                        device=device,
                    )

                    # compute loss
                    loss_video_target = loss["loss_video"](
                        logits_per_video, ground_truth
                    )
                    loss_text_target = loss["loss_text"](logits_per_text, ground_truth)
                    total_loss_target = (loss_video_target + loss_text_target) / 2
                    total_loss += config.loss.target.weight * total_loss_target

                    # update counters
                    assert logits_per_video.size(0) == logits_per_text.size(0)
                    train_total_loss_target.update(
                        total_loss_target.item(), logits_per_video.size(0)
                    )
                    train_loss_video_target.update(
                        loss_video_target.item(), logits_per_video.size(0)
                    )
                    train_loss_text_target.update(
                        loss_text_target.item(), logits_per_video.size(0)
                    )

        # backward loss
        total_loss.backward()

        # optimization step
        if device == "cpu":
            optimizers["main_optimizer"].step()
        else:
            convert_models_to_fp32(model["full"])
            optimizers["main_optimizer"].step()
            clip.model.convert_weights(model["full"])

    percentage_accepted_pseudo_labels = 0
    if config.loss.target.weight:
        percentage_accepted_pseudo_labels = accepted_pseudo_labels / total_pseudo_labels
        percentage_accepted_pseudo_labels *= 100

    if config.logging.comet:
        experiment.log_metric(
            "train_total_loss_source", train_total_loss_source.avg, step=epoch
        )
        experiment.log_metric(
            "train_loss_video_source", train_total_loss_source.avg, step=epoch
        )
        experiment.log_metric(
            "train_loss_text_source", train_loss_text_source.avg, step=epoch
        )
        experiment.log_metric(
            "train_total_loss_target", train_total_loss_target.avg, step=epoch
        )
        experiment.log_metric(
            "train_loss_video_target", train_loss_video_target.avg, step=epoch
        )
        experiment.log_metric(
            "train_loss_text_target", train_loss_text_target.avg, step=epoch
        )
        experiment.log_metric(
            "entropy_loss_target", entropy_loss_target.avg, step=epoch
        )
        experiment.log_metric("pseudo_labels_acc", pseudo_label_acc.avg, step=epoch)
        experiment.log_metric(
            "accepted_pseudo_labels(%)", percentage_accepted_pseudo_labels, step=epoch
        )
        if config.open_set.method == "attributes":
            experiment.log_metric(
                "open_set_labels",
                len(matching_results["open_set_labels"]),
                step=epoch,
            )

    train_total_loss_source.reset()
    train_loss_video_source.reset()
    train_loss_text_source.reset()
    train_total_loss_target.reset()
    train_loss_video_target.reset()
    train_loss_text_target.reset()
    train_total_loss_domain.reset()
    train_loss_video_domain.reset()
    train_loss_text_domain.reset()
    entropy_loss_target.reset()
    consistency_loss_target.reset()
    pseudo_label_acc.reset()

    for label in new_samples_per_pseudo_labels:
        new_samples_per_pseudo_labels[label] = [
            index
            for index, _ in sorted(
                new_samples_per_pseudo_labels[label], key=lambda x: x[1], reverse=True
            )
        ]

    res = {"samples_per_pseudo_labels": new_samples_per_pseudo_labels}
    if config.open_set.method == "autolabel":
        res["open_set_labels"] = matching_results["open_set_labels"]
    elif config.open_set.method == "oracle":
        res["open_set_labels"] = prompts["open_set_labels"]

    return res
