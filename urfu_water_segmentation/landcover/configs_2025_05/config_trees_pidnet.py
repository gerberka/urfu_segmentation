_base_ = [
    '../_base_/default_runtime.py'
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License

# ====== Общие настройки ======

dataset_type = 'TreesDataset'
data_root = '/misc/home6/m_imm_freedata/Segmentation/Trees/Trees_DFC_512'
classes = ('background', 'tree')
palette = [(0, 0, 0), (0, 255, 0)]

crop_size = (512, 512)
# Поскольку мы используем только один графический процессор, вместо SyncBN используется BN (при необходимости, наоборот)
norm_cfg = dict(type='BN', requires_grad=True)  # 1 GPU → BN
pretrained_checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa

# ====== Модель ======
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_checkpoint)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=2, # `background`` and `tree`
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            # OHEM (под 512x512 уменьшаем min_kept)
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=65536,  # ~1/4 от 512*512
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=5.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ====== Пайплайны ======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        # масштабы вокруг 512 со "склеиванием" по короткой стороне
        scales=[int(512 * x * 0.1) for x in range(6, 17)],  # 0.6x..1.6x
        resize_type='ResizeShortestEdge',
        max_size=1536
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4), # для BoundaryLoss
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# ====== Даталоадеры ======
batch_size = 16
num_workers = 8

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/gt'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes, palette=palette)
        # при необходимости: ann_file='splits/train.txt'
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/gt'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes, palette=palette)
        # при необходимости: ann_file='splits/val.txt'
    )
)

test_dataloader = val_dataloader

# ====== Оптимизатор и шедулер ======

# optimizer
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    # включи аккумуляцию, если нужно эмулировать большой батч
    # accumulative_counts=8,  # включи при необходимости
)


# learning policy
max_epochs = 100
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=max_epochs,
        by_epoch=True
    )
]

# training schedule for 120k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ====== Метрики и визуализация ======
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

experiment_name = f'PIDNetS_Trees512_CE_OHEM_Boundary_b{batch_size}_adamw'
work_dir = f'logs/{experiment_name}'

vis_backends = [
    dict(type='LocalVisBackend', scalar_save_file='scalars.json', save_dir=work_dir),
    dict(type='TensorboardVisBackend', save_dir=work_dir)
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ====== Хуки ======
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
