from isegm.utils.exp_imports.default import *
from isegm.data.datasets.ade20k import ADE20kDataset
from isegm.data.datasets.saliency import SaliencyDataset
from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
import torch.nn as nn
from isegm.engine.cdnet_trainer import ISTrainer
from isegm.data.aligned_augmentation import AlignedAugmentator

label = "arteria_mesenterica_superior"
MODEL_NAME = f'cdnet_res34_{label}'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (96, 96)
    model_cfg.num_max_points = 24
    model = DeeplabModel(backbone='resnet34', deeplab_ch=128, aspp_dropout=0.20, use_leaky_relu=True,
                         use_rgb_conv=False, use_disks=True, norm_radius=1, with_prev_mask=True)
    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights()
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1
    loss_cfg.fdm_instances_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2, from_sigmoid=True)  # SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    loss_cfg.fdm_instances_loss_weight = 0.4
    loss_cfg.diversity_loss = DiversityLoss()
    loss_cfg.diversity_loss_weight = 1

    train_augmentator = AlignedAugmentator(ratio=[0.3, 1.3], target_size=crop_size, flip=True,
                                           distribution='Gaussian', gs_center=0.8, gs_sd=0.4
                                           )

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       use_hierarchy=False,
                                       first_click_center=True)

    trainset = PancDataset(
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        label=label
    )

    valset = PancDataset(
        split='val',
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500,
        label=label
    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[80, 105], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 50), (100, 5)],
                        image_dump_interval=500,
                        metrics=[AdaptiveIoU(), DiceScore()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=0)
    trainer.run(num_epochs=220)
