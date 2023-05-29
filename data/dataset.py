import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
from PIL import Image
from random import Random
from .data_utils import natural_keys

from os import listdir
from os.path import join


class VideoDataset(data.Dataset):
    def __init__(
        self,
        dataset_input,
        root,
        num_segments=1,
        new_length=1,
        frame_tmpl="{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        index_bias=1,
        epic_kitchens=False,
        slurm=False,
        alderaan=False,
        hpc=False,
        open_set=False,
        return_paths=False,
    ):

        self.root = root
        self.num_segments = num_segments
        self.seg_length = new_length
        self.frame_tmpl = frame_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.epic_kitchens = epic_kitchens
        self.slurm = slurm
        self.alderaan = alderaan
        self.hpc = hpc
        self.open_set = open_set
        self.return_paths = return_paths

        if self.index_bias is None:
            if self.frame_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1

        self.classes = []
        self.video_list = []

        if self.epic_kitchens:
            self.ek_videos = {}
        self.parse_file(dataset_input)

        self.initialized = False

    def parse_file(self, dataset_input):
        with open(dataset_input, "r") as file_list:
            i = 0
            for line in file_list:
                split_line = line.split()
                path = split_line[0]
                path = join(self.root, path)
                if self.epic_kitchens:
                    start_frame = int(split_line[1])
                    stop_frame = int(split_line[2])
                    label = int(split_line[3])
                    self.video_list.append((path, start_frame, stop_frame, label, i))
                    kitchen = path.split("/")[-1]
                    if kitchen not in self.ek_videos:
                        kitchen_videos = self.find_frames(path)
                        kitchen_videos.sort(key=natural_keys)
                        self.ek_videos[kitchen] = kitchen_videos
                else:
                    label = int(split_line[1])
                    self.video_list.append((path, label, i))
                i += 1

    def _load_image(self, directory, idx):

        return [
            Image.open(os.path.join(directory, self.frame_tmpl.format(idx))).convert(
                "RGB"
            )
        ]

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    # selects frames from input sequence
    def find_frames(self, video):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # checks if input is image
    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    def _sample_indices(self, video):
        offsets = list()

        num_segments = self.num_segments

        if self.epic_kitchens:
            kitchen = video["video"].split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[video["start_frame"] : video["stop_frame"]]
            length = len(frame_paths)
        else:
            length = len(self.find_frames(video))

        ticks = [i * length // num_segments for i in range(num_segments + 1)]

        for i in range(num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, video):

        num_segments = self.num_segments

        if self.epic_kitchens:
            kitchen = video["video"].split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[video["start_frame"] : video["stop_frame"]]
            length = len(frame_paths)
        else:
            length = len(self.find_frames(video))
        if num_segments == 1:
            return np.array([length // 2], dtype=np.int) + self.index_bias

        if length <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), length) + self.index_bias
            return (
                np.array(
                    [i * length // self.total_length for i in range(self.total_length)],
                    dtype=np.int,
                )
                + self.index_bias
            )
        offset = (length / num_segments - self.seg_length) / 2.0
        return (
            np.array(
                [
                    i * length / num_segments + offset + j
                    for i in range(num_segments)
                    for j in range(self.seg_length)
                ],
                dtype=int,
            )
            + self.index_bias
        )

    def get_video_2(self, label1):
        label2 = label1
        video2 = None

        while label2 == label1:
            index = randint(0, self.__len__())
            if self.epic_kitchens:
                video, start_frame, stop_frame, label2 = self.video_list[index]
                video2 = {
                    "video": video,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                }
            else:
                video2, label2 = self.video_list[index]

        assert video2 is not None
        assert label2 != label1

        return video2, label2

    def __getitem__(self, index):

        if self.epic_kitchens:
            video, start_frame, stop_frame, label, video_id = self.video_list[index]
            video = {
                "video": video,
                "start_frame": start_frame,
                "stop_frame": stop_frame,
            }
        else:
            video, label, video_id = self.video_list[index]

        segment_indices = (
            self._sample_indices(video)
            if self.random_shift
            else self._get_val_indices(video)
        )

        return self.get(video, label, segment_indices, video_id)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, video, label, indices, video_id):
        images = list()

        # find frames
        if self.epic_kitchens:
            path = video["video"]
            kitchen = path.split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[video["start_frame"] : video["stop_frame"]]
        else:
            path = video
            frame_paths = self.find_frames(video)
            frame_paths.sort(key=natural_keys)

        for i, seg_ind in enumerate(indices):
            p = int(seg_ind) - 1
            try:
                seg_imgs = [Image.open(frame_paths[p]).convert("RGB")]
            except OSError:
                print('ERROR: Could not read image "{}"'.format(video))
                print("invalid indices: {}".format(indices))
                raise
            images.extend(seg_imgs)

        process_data = self.transform(images)

        if self.return_paths:
            if self.epic_kitchens:
                return (
                    process_data,
                    label,
                    path,
                    video["start_frame"],
                    video["stop_frame"],
                    video_id,
                )
            return process_data, label, path, video_id

        return process_data, label, video_id

    def __len__(self):
        return len(self.video_list)


class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return (*source_data, *target_data, target_index)
