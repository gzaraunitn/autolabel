import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import warnings
import clip
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.utils import LabelsManager
from PIL import UnidentifiedImageError, Image
from termcolor import colored
from json import load


def test_step(
    model,
    loader,
    config,
    prompts,
    device,
    classes_names,
    epoch,
    run_id,
    training_results=None,
    sanity_check=False,
    run=None,
    experiment=None,
):

    warnings.simplefilter("ignore", ResourceWarning)

    # set models in eval mode
    for m in model:
        model[m].eval()

    # define counters
    num = 0
    corr_1 = 0
    corr_5 = 0

    num_classes = len(classes_names)

    loader = loader["val"]
    if sanity_check:
        print(colored("Sanity check...", "green"))
    else:
        print("[Testing] - Epoch {}".format(epoch + 1))
        loader = tqdm(loader)

    with torch.no_grad():

        classes = prompts["classes"]

        # text encoding
        if not sanity_check:
            if config.open_set.method in ["autolabel", "oracle"]:
                text_dict = {}
                for i, txt in enumerate(prompts["text_aug"]):
                    text_dict[i] = torch.cat(
                        [clip.tokenize(txt.format(c)) for _, c in classes_names]
                    )
                    if len(training_results["open_set_labels"]):
                        text_dict[i] = torch.cat(
                            (
                                text_dict[i],
                                torch.cat(
                                    [
                                        clip.tokenize(open_set_label)
                                        for open_set_label in training_results[
                                            "open_set_labels"
                                        ]
                                    ]
                                ),
                            )
                        )
                classes = torch.cat([v for _, v in text_dict.items()])
                extended_classes_names = [c for _, c in classes_names] + training_results["open_set_labels"]

        text_inputs = classes.to(device)
        text_features = model["text_model"](text_inputs)

        # keep track of predicted and gt labels
        pred = []
        gt = []
        ids = []
        tps = []

        # class-wise metrics
        correct_per_class = [0 for _ in range(num_classes + 1)]
        instances_per_class = [0 for _ in range(num_classes + 1)]

        attributes_file_path = "utils/attributes/{}_test_{}_vilt.json".format(
            config.data.target_dataset, config.attributes.n_blanks
        )

        if "olympics" in config.data.dataset:
            attributes_file_path = attributes_file_path.replace(
                "_test_", "_test_ucfolympics_"
            )
        if config.data.clean_ek:
            attribute_file_path = attributes_file_path.replace("_test_", "_test_clean_")

        for batch_idx, batch in enumerate(loader):
            video, y, video_id = batch

            # compute video features
            video = video.view((-1, config.data.num_segments, 3) + video.size()[-2:])
            b, t, c, h, w = video.size()
            y = y.to(device)
            video_input = video.to(device).view(-1, c, h, w)
            video_features = model["video_model"](video_input).view(b, t, -1)
            video_features = model["fusion_model"](video_features)

            if config.open_set.method == "zoc":
                assert b == 1, "Val batch size must be 1!"
                text_dict = {}
                for i, txt in enumerate(prompts["text_aug"]):
                    text_dict[i] = torch.cat(
                        [clip.tokenize(txt.format(c)) for _, c in classes_names]
                    )
                    with open(attributes_file_path, "r") as attributes_file:
                        open_set_labels = load(attributes_file)[str(video_id.item())]
                        text_dict[i] = torch.cat(
                            (
                                text_dict[i],
                                torch.cat(
                                    [
                                        clip.tokenize(open_set_label)
                                        for open_set_label in open_set_labels
                                    ]
                                ),
                            )
                        )
                        classes = torch.cat([v for _, v in text_dict.items()])
                        text_inputs = classes.to(device)
                        text_features = model["text_model"](text_inputs)

            # normalize
            video_features /= video_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # compute similarity
            similarity = 100.0 * video_features @ text_features.T

            num_text_aug = prompts["num_text_aug"]
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)

            # compute metrics
            values_1, indices_1 = similarity.topk(1, dim=-1)
            if "olympics" not in config.data.dataset:
                values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b

            # open set protocol
            if config.open_set.method in ["autolabel", "oracle", "zoc"]:
                predicted_indices = indices_1.clone()
                indices_1[indices_1 >= num_classes] = num_classes
            elif config.open_set.method == "osvm":
                indices_1[values_1 < config.open_set.threshold] = num_classes
            else:
                raise ValueError("Open set method not recognized!")

            # tracking performance
            for i in range(b):
                label = y[i]
                if indices_1[i] == label:
                    corr_1 += 1
                if "olympics" not in config.data.dataset:
                    if y[i] in indices_5[i]:
                        corr_5 += 1
                predicted_label = indices_1[i]
                instances_per_class[label] += 1
                if label.item() == predicted_label.item():
                    correct_per_class[label] += 1
            pred.extend([i[0] for i in indices_1.cpu().tolist()])
            gt.extend(list(y.cpu().tolist()))
            ids.extend(list(video_id.cpu().tolist()))
            # tps.extend([i[0] for i in predicted_indices.cpu().tolist()])

            if sanity_check:
                if batch_idx >= config.general.sanity_check_steps:
                    break

    acc1 = float(corr_1) / num * 100
    if "olympics" not in config.data.dataset:
        acc5 = float(corr_5) / num * 100

    # open set metrics
    h_score = 0.0
    if not sanity_check:
        accuracy_per_class = np.array(correct_per_class) / np.array(instances_per_class)
        closed_accuracy = accuracy_per_class[:num_classes].mean()
        open_accuracy = accuracy_per_class[-1]
        h_score = (
            2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)
        )

    # log metrics
    if not sanity_check:

        labels_manager = LabelsManager(config)

        # confusion matrix
        labels = [i[1] for i in classes_names]
        labels.append("UNK")
        cm = confusion_matrix(
            labels_manager.convert(gt),
            labels_manager.convert(pred),
            labels=labels,
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation="vertical")
        plt.savefig("cm.png")
        plt.close()

        if config.logging.comet:
            experiment.log_metric("validation_accuracy", acc1, step=epoch)
            if "olympics" not in config.data.dataset:
                experiment.log_metric("acc5", acc5, step=epoch)
            experiment.log_confusion_matrix(
                gt,
                pred,
                labels=labels,
            )
            experiment.log_image(Image.open("cm.png"), name="cm_epoch={}".format(epoch))
            experiment.log_metric("closed_accuracy", closed_accuracy, step=epoch)
            experiment.log_metric("open_accuracy", open_accuracy, step=epoch)
            experiment.log_metric("h_score", h_score, step=epoch)

        if "olympics" not in config.data.dataset:
            line = (
                "Epoch: [{}/{}]:\n  VAL ACCURACY @1: {}\n  VAL ACCURACY @5: {}".format(
                    epoch + 1, config.solver.epochs, acc1, acc5
                )
            )
        else:
            line = "Epoch: [{}/{}]:\n  VAL ACCURACY @1: {}\n".format(
                epoch + 1, config.solver.epochs, acc1
            )
        line += (
            "\n  CLOSED ACCURACY: {}\n  OPEN ACCURACY: {}\n  H-SCORE: {}".format(
                closed_accuracy, open_accuracy, h_score
            )
        )
        print(line)
        with open(config.log_file_path, "a") as logfile:
            logfile.write("{}\n\n".format(line))

    res = h_score

    return res
