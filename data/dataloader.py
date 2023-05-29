from .dataset import VideoDataset, VideoDatasetSourceAndTarget
from torch.utils.data import DataLoader
from os.path import join


# create datasets
def get_datasets(config, training_transforms, test_transforms):

    epic_kitchens = "ek" in config.data.dataset

    # edit with your own path
    if epic_kitchens:
        source_root = target_root = val_root = config.data.root
    else:
        source_root = join(config.data.root, config.data.source_dataset, "train")
        target_root = join(config.data.root, config.data.target_dataset, "train")
        val_root = join(config.data.root, config.data.target_dataset, "test")

    source_train_input = config.data.source_train_file
    target_train_input = config.data.target_train_file
    test_input = config.data.test_file

    source_train_dataset = VideoDataset(
        source_train_input,
        root=source_root,
        num_segments=config.data.num_segments,
        frame_tmpl=config.data.frame_tmpl,
        random_shift=config.data.random_shift,
        transform=training_transforms,
        epic_kitchens=epic_kitchens,
        slurm=config.general.slurm,
        alderaan=config.general.alderaan,
        hpc=config.general.hpc,
    )

    target_train_dataset = VideoDataset(
        target_train_input,
        root=target_root,
        num_segments=config.data.num_segments,
        frame_tmpl=config.data.frame_tmpl,
        random_shift=config.data.random_shift,
        transform=training_transforms,
        epic_kitchens=epic_kitchens,
        slurm=config.general.slurm,
        alderaan=config.general.alderaan,
        hpc=config.general.hpc,
        return_paths=True,
    )

    train_dataset = VideoDatasetSourceAndTarget(
        source_train_dataset, target_train_dataset
    )

    val_dataset = VideoDataset(
        test_input,
        root=val_root,
        random_shift=False,
        num_segments=config.data.num_segments,
        frame_tmpl=config.data.frame_tmpl,
        transform=test_transforms,
        epic_kitchens=epic_kitchens,
        slurm=config.general.slurm,
        alderaan=config.general.alderaan,
        hpc=config.general.hpc,
    )

    if config.loss.target.filtering == "use_class_wise_thresholds":
        return train_dataset, val_dataset, source_train_dataset, target_train_dataset

    return train_dataset, val_dataset, source_train_dataset, target_train_dataset


# create dataloaders
def get_dataloaders(config, training_transforms, test_transforms):

    train_dataset, val_dataset, source_dataset, target_dataset = get_datasets(
        config=config,
        training_transforms=training_transforms,
        test_transforms=test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    source_train_loader = DataLoader(
        source_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    target_train_loader = DataLoader(
        target_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    if config.open_set.method == "zoc":
        val_batch_size = 1
    else:
        val_batch_size = config.data.batch_size

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=config.data.workers,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    res = {
        "train": train_loader,
        "val": val_loader,
        "source": source_train_loader,
        "target": target_train_loader,
    }

    return res
