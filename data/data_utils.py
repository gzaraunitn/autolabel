import re
from torchvision import transforms
from .custom_transforms import *
from os import listdir
from RandAugment import RandAugment
from csv import reader


# converts string to integers when possible
def atoi(text):
    return int(text) if text.isdigit() else text


# applies atoi to a string
def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


# training transformations
def get_training_transforms(config):
    # TODO: find correct mean and std
    trs = transforms.Compose(
        [
            GroupMultiScaleCrop(config.data.input_size, [1, 0.875, 0.75, 0.66]),
            GroupRandomHorizontalFlip(),
            GroupRandomColorJitter(
                p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            ),
            GroupRandomGrayscale(p=0.2),
            GroupGaussianBlur(p=0.0),
            GroupSolarization(p=0.0),
        ]
    )
    return trs


# test transformations
def get_test_transforms(config):
    scale_size = config.data.input_size * 256 // 224
    trs = transforms.Compose(
        [GroupScale(scale_size), GroupCenterCrop(config.data.input_size)]
    )
    return trs


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


# rand-augment transformation
def rand_augment(transform_train, config):
    transform_train.transforms.insert(
        0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M))
    )
    return transform_train


# total transforms
def get_transforms(config):
    # TODO: find correct mean and std
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    training_transforms = get_training_transforms(config)
    test_transforms = get_test_transforms(config)
    common_transforms = transforms.Compose(
        [
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]
    )
    training_transforms = transforms.Compose([training_transforms, common_transforms])
    test_transforms = transforms.Compose([test_transforms, common_transforms])
    return training_transforms, test_transforms


def get_classes(config):

    ek_map = {
        "opening": 2,
        "taking": 0,
        "closing": 3,
        "putting": 1,
        "washing": 4,
        "pouring": 7,
        "mixing": 6,
        "cutting": 5,
    }

    ek_map2 = {
        "mixing": 2,
        "closing": 0,
        "moving": 3,
        "inserting": 1,
        "opening": 4,
        "washing": 7,
        "taking": 6,
        "putting": 5,
    }

    hu_map = {
        "climb": 0,
        "fencing": 1,
        "golf": 2,
        "kick ball": 3,
        "pullup": 4,
        "punch": 5,
    }

    if "ek" in config.data.dataset:
        # res = [[i, c] for c, i in ek_map.items()]
        res = [[i, c] for c, i in sorted(ek_map.items(), key=lambda x:x[1])]
    else:
        res = [[i, c] for c, i in hu_map.items()]

    return res


def get_open_set_classes(config):
    res = []
    if "ek" in config.data.dataset:
        file_paths = ["data/EPIC_100_train.csv", "data/EPIC_100_validation.csv"]
        with open(config.data.test_file, "r") as test_file:
            for line in test_file:
                path, start_frame, stop_frame, label = line.split()
                video_id = path.split("/")[-1]
                start_frame, stop_frame = line.split()[1:3]
                if int(label) == len(get_classes(config)):
                    for fpath in file_paths:
                        with open(fpath, "r") as csv_file:
                            csv_reader = reader(csv_file, delimiter=",")
                            count = 0
                            for line in csv_reader:
                                if count == 0:
                                    fields = line
                                else:
                                    candidate_video_id = line[fields.index("video_id")]
                                    if candidate_video_id == video_id:
                                        candidate_start_frame = line[
                                            fields.index("start_frame")
                                        ]
                                        candidate_stop_frame = line[
                                            fields.index("stop_frame")
                                        ]
                                        cond1 = int(candidate_start_frame) == int(start_frame)
                                        cond2 = int(candidate_stop_frame) == int(stop_frame)
                                        if cond1 and cond2:
                                            class_name = line[fields.index("verb")]
                                            if class_name not in res:
                                                res.append(class_name)
                                            break

                                count += 1
    else:
        with open(config.data.test_file, "r") as test_file:
            for line in test_file:
                path, label = line.split()
                if int(label) == len(get_classes(config)):
                    class_name = path.split("/")[0]
                    if class_name not in res:
                        res.append(class_name)
    assert len(res)
    return res


def find_n_classes(txt_file):
    labels = []
    with open(txt_file, "r") as txtfile:
        for line in txtfile:
            split_line = line.split()
            label = split_line[-1]
            labels.append(int(label))
    return len(list(set(labels)))
