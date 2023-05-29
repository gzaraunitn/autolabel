import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from utils.utils import LabelsManager
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from json import load
from random import choice
from yellowbrick.cluster import KElbowVisualizer


def k_means(model, loader, config, device):

    res = {}

    # set models in eval mode
    for m in model:
        model[m].eval()

    labels_manager = LabelsManager(config)

    feats = []
    ids = []
    true_labels = []
    with torch.no_grad():
        print("K-means: loading training set")
        if "ek" in config.data.dataset:
            start_frames = []
            stop_frames = []
            for batch_idx, (
                video_target,
                y_target,
                path,
                start_frame,
                stop_frame,
                video_id,
            ) in enumerate(tqdm(loader)):

                # reshape the source input and get size
                video_target = video_target.view(
                    (-1, config.data.num_segments, 3) + video_target.size()[-2:]
                )
                b, t, c, h, w = video_target.size()

                # move to gpu device
                video_target = video_target.to(device).view(-1, c, h, w)

                # produce embeddings
                video_embedding = model["video_model"](video_target)
                video_embedding = video_embedding.view(b, t, -1)
                video_embedding = model["fusion_model"](video_embedding)
                for i in range(b):
                    feats.append(
                        video_embedding[i]
                        .reshape(video_embedding.size(-1))
                        .cpu()
                        .numpy()
                    )
                    start_frames.append(start_frame[i].item())
                    stop_frames.append(stop_frame[i].item())
                    ids.append(video_id[i].item())
                    true_labels.append(
                        labels_manager.convert_single_label(y_target[i].item())
                    )
            res["start_frames"] = start_frames
            res["stop_frames"] = stop_frames
        else:
            for batch_idx, (video_target, y_target, path, video_id) in enumerate(
                tqdm(loader)
            ):

                # reshape the source input and get size
                video_target = video_target.view(
                    (-1, config.data.num_segments, 3) + video_target.size()[-2:]
                )
                b, t, c, h, w = video_target.size()

                # move to gpu device
                video_target = video_target.to(device).view(-1, c, h, w)

                # produce embeddings
                video_embedding = model["video_model"](video_target)
                video_embedding = video_embedding.view(b, t, -1)
                video_embedding = model["fusion_model"](video_embedding)
                for i in range(b):
                    feats.append(
                        video_embedding[i]
                        .reshape(video_embedding.size(-1))
                        .cpu()
                        .numpy()
                    )
                    ids.append(video_id[i].item())
                    true_labels.append(
                        labels_manager.convert_single_label(y_target[i].item())
                    )

    print("Performing clustering...")

    if config.attributes.use_elbow:
        print("Using elbow method to find optimal K")
        km = KMeans()
        visualizer = KElbowVisualizer(km, k=(4, 50))
        visualizer.fit(np.array(feats))
        k = visualizer.elbow_value_
        print("K = {}".format(k))
    else:
        k = config.attributes.k_clustering

    res["labels"] = KMeans(
        n_clusters=k, random_state=0
    ).fit_predict(feats)
    res["true_labels"] = true_labels
    print("Clustering completed")
    res["ids"] = ids
    res["k"] = k

    # set models in train mode
    for m in model:
        model[m].train()

    return res


def extract(res, caption):
    c = caption.split(" ")
    indices = np.where(np.array(c) == "[MASK]")[0]
    return "_".join([res.split()[index] for index in indices])


def tf_idf(most_commons):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(most_commons)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names).T
    return df


def get_most_commons(input_dict, config):
    most_commons = []
    for label in sorted(input_dict, key=lambda x: int(x)):
        counter = Counter(input_dict[label])
        most_common = " ".join(
            [word for word, _ in counter.most_common(config.attributes.n_attributes)]
        )
        most_commons.append(most_common)
    return most_commons


def extract_cluster_attributes(clustering_results, config, epoch, run_id, elbow=None):
    attribute_file_path = "utils/attributes/{}_target_{}.json".format(
        config.data.target_dataset, config.attributes.n_blanks
    )

    with open(attribute_file_path, "r") as attribute_file:
        attributes = load(attribute_file)
    attributes_per_cluster_label = defaultdict(list)
    true_labels_per_cluster = defaultdict(list)

    for i in range(len(clustering_results["labels"])):
        cluster_label = clustering_results["labels"][i]
        for attr in attributes[str(clustering_results["ids"][i])]:
            attributes_per_cluster_label[cluster_label].append(attr)
            true_labels_per_cluster[str(cluster_label)].append(
                clustering_results["true_labels"][i]
            )

    most_commons = get_most_commons(attributes_per_cluster_label, config)
    df = tf_idf(most_commons)
    attributes_per_cluster = {}
    for i in range(clustering_results["k"]):
        if config.attributes.selection == "topk":
            attributes = (
                df[i].nlargest(config.attributes.tf_idf_topk_target).index.tolist()
            )
        elif config.attributes.selection == "threshold":
            attributes = df[df[i] > config.attributes.tf_idf_threshold].index.tolist()
        else:
            raise ValueError(
                "Attribute selection strategy {} not recognized!".format(
                    config.attributes.selection
                )
            )
        attributes_per_cluster[i] = attributes

        with open("{}_attributes.txt".format(run_id), "a") as attributes_file:
            attributes_file.write(
                "\n\n{}: original: {}\n, middle: {}\n, final: {}\n".format(
                    epoch,
                    attributes_per_cluster_label,
                    most_commons,
                    attributes_per_cluster,
                )
            )

    generic_true_labels_per_cluster = defaultdict(dict)
    for cluster_label in true_labels_per_cluster:
        inner_dict = {}
        labels_set = list(set(true_labels_per_cluster[cluster_label]))
        for label in labels_set:
            inner_dict[label] = true_labels_per_cluster[cluster_label].count(label)
        generic_true_labels_per_cluster[cluster_label] = inner_dict

    return attributes_per_cluster, generic_true_labels_per_cluster


def extract_class_attributes(config):

    labels_manager = LabelsManager(config)

    attribute_file_path = "utils/attributes/{}_source_{}.json".format(
        config.data.source_dataset, config.attributes.n_blanks
    )
    with open(attribute_file_path, "r") as attribute_file:
        attributes = load(attribute_file)

    attributes_per_class = defaultdict(list)
    with open(config.data.source_train_file, "r") as txt_file:
        epic_kitchens = "ek" in config.data.dataset
        for i, line in enumerate(txt_file):
            if epic_kitchens:
                _, _, _, label = line.split()
            else:
                _, label = line.split()
            for attr in attributes[str(i)]:
                attributes_per_class[label].append(attr)
        most_commons = get_most_commons(attributes_per_class, config)
        df = tf_idf(most_commons)
        clean_attributes_per_class = {}
        for i in range(len(attributes_per_class.keys())):
            if config.attributes.selection == "topk":
                attributes = (
                    df[i].nlargest(config.attributes.tf_idf_topk_source).index.tolist()
                )
            elif config.attributes.selection == "threshold":
                attributes = df[
                    df[i] > config.attributes.tf_idf_threshold
                ].index.tolist()
            else:
                raise ValueError(
                    "Attribute selection strategy {} not recognized!".format(
                        config.attributes.selection
                    )
                )
            clean_attributes_per_class[
                labels_manager.convert_single_label(i)
            ] = attributes

    return clean_attributes_per_class


def compute_score(s, t, weights):
    if s == t:
        return 1
    else:
        return 1 * weights[abs(t - s)]


def match_attributes(
    source_attributes_per_class, target_attributes_per_cluster, config
):
    matches_per_cluster_label = {}
    for cluster_label in target_attributes_per_cluster:
        target_attributes = target_attributes_per_cluster[cluster_label]

        matched_source_labels = []
        for source_label in source_attributes_per_class:
            source_attributes = source_attributes_per_class[source_label]

            ref = np.flip(np.array(list(range(len(source_attributes)))))
            weights = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
            score = 0
            for s in range(len(source_attributes)):
                for t in range(len(target_attributes)):
                    if target_attributes[t] == source_attributes[s]:
                        score += weights[abs(t - s)]

            score /= len(source_attributes)

            match = score > config.attributes.matching_threshold
            if match:
                if config.logging.verbose:
                    print("------------------------------------------")
                    print("MATCH: score = {}".format(score))
                    print("SA({}) = {}".format(source_label, source_attributes))
                    print("TA = {}".format(target_attributes))
                    print("------------------------------------------")
                matched_source_labels.append((source_label, score))
        if len(matched_source_labels):
            max_confidence = max([conf for _, conf in matched_source_labels])
            top_confident_labels = [
                label for label, conf in matched_source_labels if conf == max_confidence
            ]
            if len(top_confident_labels) > 1:
                candidate_label = choice(top_confident_labels)
            else:
                candidate_label = top_confident_labels[0]
            matches_per_cluster_label[cluster_label] = candidate_label

    open_set_labels = []
    matched_cluster_labels = []
    unmatched_cluster_labels = []
    for cluster_label in target_attributes_per_cluster:
        if cluster_label not in matches_per_cluster_label:
            open_set_label = " and ".join(
                target_attributes_per_cluster[cluster_label][
                    : config.attributes.final_prompt_length
                ]
            )
            matches_per_cluster_label[cluster_label] = open_set_label
            if open_set_label not in open_set_labels:
                open_set_labels.append(open_set_label)
                unmatched_cluster_labels.append(cluster_label)
        else:
            matched_cluster_labels.append(cluster_label)

    res = {
        "matches_per_cluster_label": matches_per_cluster_label,
        "open_set_labels": open_set_labels,
        "matched_cluster_labels": matched_cluster_labels,
        "unmatched_cluster_labels": unmatched_cluster_labels,
    }

    return res


def compute_clustering_accuracy(
    matches_per_cluster_label, clustering_labels, true_labels, config
):
    labels_manager = LabelsManager(config)

    pred = []
    for i in range(len(clustering_labels)):
        assigned_label = clustering_labels[i]
        found = False
        for cluster_label in matches_per_cluster_label:
            if assigned_label == cluster_label:
                if assigned_label in labels_manager.rev_label_map:
                    pred.append(
                        labels_manager.convert_single_label(
                            assigned_label, reverse=True
                        )
                    )
                else:
                    pred.append(
                        labels_manager.convert_single_label("UNK", reverse=True)
                    )
                found = True
        assert found
    assert len(pred) == len(clustering_labels)
    true = [labels_manager.convert_single_label(l, reverse=True) for l in true_labels]
    acc = accuracy_score(pred, true)
    return acc
