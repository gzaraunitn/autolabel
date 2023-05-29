import numpy as np
import torch

from collections import defaultdict
from os import listdir
from tqdm import tqdm
from random import choice
from string import ascii_lowercase, digits


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt


def compute_acc(y, y_hat):
    """Computes accuracy."""

    acc = y == y_hat
    if len(acc) > 0:
        acc = acc.sum().detach().true_divide(acc.size(0))
    else:
        acc = torch.tensor(0.0, device=y.device)

    return acc



def compute_accept_mask(
    model,
    video_embedding,
    classes,
    prompts,
    device,
    config,
    thresholds,
    samples_per_pseudo_labels,
    indices,
):
    with torch.no_grad():

        # compute text features
        text_inputs = classes.to(device)
        text_features = model["text_model"](text_inputs)

        # normalize
        video_features = video_embedding / video_embedding.norm(dim=-1, keepdim=True)
        b = video_features.size(0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        video_features = video_features.float()
        text_features = text_features.float()

        # compute similarity
        similarity = 100.0 * video_features @ text_features.T
        similarity = similarity.view(video_features.size(0), prompts["num_text_aug"], -1).softmax(dim=-1)
        similarity = similarity.mean(dim=1, keepdim=False)

        # compute metrics
        values, pseudo_y_target = similarity.topk(1, dim=-1)
        pseudo_y_target = pseudo_y_target.view(pseudo_y_target.size(0))
        values = values.view(values.size(0))

        # compute mask
        if config.loss.target.use_gt:
            mask = torch.ones(b, dtype=torch.bool, device=device)
        else:
            if config.loss.target.filtering == "class_wise_thresholds":
                threshold_values = torch.tensor(
                    [thresholds[i.item()] for i in pseudo_y_target]
                ).to(device)
                mask = (values > threshold_values).view(b)
            elif config.loss.target.filtering == "top_k_confident_samples":
                if samples_per_pseudo_labels is not None:
                    mask = []
                    for i in range(b):
                        pseudo_label = pseudo_y_target[i].item()
                        total_length = len(samples_per_pseudo_labels[pseudo_label])
                        k = config.loss.target.k
                        if config.loss.target.k_type == "percentage":
                            k = int((total_length / 100) * config.loss.target.k)
                        cond = (
                            indices[i].item()
                            in samples_per_pseudo_labels[pseudo_y_target[i].item()][:k]
                        )
                        mask.append(bool(cond))
                    mask = torch.tensor(mask, dtype=torch.bool, device=device)
                else:
                    mask = torch.zeros(b, dtype=torch.bool, device=device)
            elif config.loss.target.filtering == "single_threshold":
                mask = (values > config.loss.target.confidence_threshold).view(b)
            else:
                print("Filtering method not recognized!")
                exit(1)
        accepted = torch.sum(mask)

        if config.open_set.method == "aaai":
            shared_mask = torch.zeros_like(mask)
            shared_mask[pseudo_y_target < len(prompts["classes_names"])] = 1
            mask = torch.logical_and(mask, shared_mask)

        return mask, accepted, pseudo_y_target, values


def get_random_string(length):
    # choose from all lowercase letter
    characters = ascii_lowercase + digits
    result_str = "".join(choice(characters) for _ in range(length))
    return result_str


def process_run_name(config, id_len=8):
    run_name = config.logging.run_name
    if config.loss.target.use_gt:
        run_name += "_targetGT"
    run_name += "_ODA={}".format(config.open_set.method)
    run_id = get_random_string(id_len)
    run_name_no_id = run_name
    run_name = "{}_{}".format(run_id, run_name)
    return run_name, run_id, run_name_no_id


class LabelsManager:
    def __init__(self, config):

        if "ek" in config.data.dataset:
            self.label_map = {
                2: "opening",
                0: "taking",
                3: "closing",
                1: "putting",
                4: "washing",
                7: "pouring",
                6: "mixing",
                5: "cutting",
            }
            self.label_map.update({8: "UNK"})
        else:
            self.label_map = {
                0: "climb",
                1: "fencing",
                2: "golf",
                3: "kick ball",
                4: "pullup",
                5: "punch",
            }
            self.label_map.update({6: "UNK"})

        self.rev_label_map = {v: k for k, v in self.label_map.items()}

    def convert(self, labels, reverse=False):
        if reverse:
            return [self.rev_label_map[label] for label in labels]
        return [self.label_map[label] for label in labels]

    def convert_single_label(self, label, reverse=False):
        if reverse:
            return self.rev_label_map[label]
        return self.label_map[label]

    def index_to_example(self, index):
        return self.label_map[index]


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def epoch_saving(epoch, model, fusion_model, optimizer, filename):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "fusion_model_state_dict": fusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )  # just change to your preferred folder/filename


def best_saving(working_dir, epoch, model, fusion_model, optimizer):
    best_name = "{}/model_best.pt".format(working_dir)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "fusion_model_state_dict": fusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        best_name,
    )  # just change to your preferred folder/filename


# computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

