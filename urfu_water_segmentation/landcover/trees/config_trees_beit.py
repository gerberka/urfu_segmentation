# ================================================================
# Base
# ================================================================
_base_ = '../../../configs/beit/beit-large_upernet_8xb1-amp-160k_ade20k-640x640.py'

# ================================================================
# Dataset
# ================================================================
dataset_type = 'TreesDataset'
data_root = '/misc/home6/m_imm_freedata/Segmentation/Trees/Trees_DFC_512'

num_classes = 2
crop_size = (512, 512)

# (Optional, но обычно полезно для метрик/визуализации)
metainfo = dict(
    classes=('bg', 'tree'),
    palette=[(0, 0, 0), (0, 255, 0)],
)

custom_imports = dict(
    imports=[
        # transforms
        'transforms.debug_gt',
        'transforms.ensure_single_channel_gt',
        'transforms.sanitize_binary_gt',
        # если TreesDataset кастомный и не регистрируется автоматически — добавь здесь модуль датасета
        # 'datasets.trees_dataset',
    ],
    allow_failed_imports=False,
)

# ================================================================
# Training (high-level)
# ================================================================
max_epochs = 200
batch_size = 4
num_workers = 8
log_interval = 10

# Grad accumulation
accumulative_counts = 2
effective_batch_size = batch_size * accumulative_counts  # per-GPU, если не DDP. В DDP умножь ещё на world_size.

# Optimizer / schedule params
optimizer_type = 'AdamW'
base_lr = 2e-5
weight_decay = 0.05
betas = (0.9, 0.999)

# Layer decay
layer_decay_num_layers = 24
layer_decay_rate = 0.95

# Warmup
warmup_epochs = 5
warmup_start_factor = 1e-3

# Loss / ignore
ignore_index = 255
loss_name = 'CE'

# Slide inference params
test_mode = 'slide'
slide_stride = (426, 426)

# Pretrain
checkpoint_file = 'pretrain/beit_large_patch16_224_pt22k_ft22k.pth'

# ================================================================
# Experiment name
# ================================================================
experiment_name = (
    f'beit-large_upernet'
    f'__ds=Trees_DFC_512'
    f'__classes={num_classes}'
    f'__crop={crop_size[0]}x{crop_size[1]}'
    f'__loss={loss_name}'
    f'__opt={optimizer_type}'
    f'__lr={base_lr:g}'
    f'__wd={weight_decay:g}'
    f'__betas={betas[0]:g}-{betas[1]:g}'
    f'__acc={accumulative_counts}'
    f'__bs={batch_size}'
    f'__effbs={effective_batch_size}'
    f'__ld={layer_decay_rate:g}'
    f'__ldepl={layer_decay_num_layers}'
    f'__warm={warmup_epochs}ep@{warmup_start_factor:g}'
    f'__sched=poly'
    f'__ep={max_epochs}'
    f'__test={test_mode}'
    f'__stride={slide_stride[0]}x{slide_stride[1]}'
)

logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'

# ================================================================
# Preprocessor
# ================================================================
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=ignore_index,
)

# ================================================================
# Model
# ================================================================
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # веса через init_cfg у backbone

    backbone=dict(
        type='BEiT',
        img_size=crop_size,
        patch_size=16,
        in_channels=3,

        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,

        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,

        out_indices=(7, 11, 15, 23),

        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=1e-6,

        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    ),

    neck=dict(
        type='Feature2Pyramid',
        embed_dim=1024,
        rescales=[4, 2, 1, 0.5],
    ),

    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),

    test_cfg=dict(mode='slide', crop_size=crop_size, stride=slide_stride),
)

# ================================================================
# Optimization
# ================================================================
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type=optimizer_type,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay,
    ),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=layer_decay_num_layers,
        layer_decay_rate=layer_decay_rate,
        # если хочешь “как в офф. конфигах” для трансформеров — часто добавляют no_decay для норм/pos_embed и т.п.
        # custom_keys={
        #     'pos_embed': dict(decay_mult=0.0),
        #     'cls_token': dict(decay_mult=0.0),
        #     'norm': dict(decay_mult=0.0),
        # }
    ),
    accumulative_counts=accumulative_counts,
    # clip_grad часто полезен для ViT/BEiT:
    # clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=warmup_start_factor,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
    ),
    dict(
        type='PolyLR',
        power=1.0,
        begin=warmup_epochs,
        end=max_epochs,
        eta_min=0.0,
        by_epoch=True,
    ),
]

# ================================================================
# Pipelines
# ================================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='EnsureSingleChannelGT'),
    dict(type='SanitizeBinaryGT'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='EnsureSingleChannelGT'),
    dict(type='SanitizeBinaryGT'),
    dict(type='PackSegInputs'),
]

# ================================================================
# Dataloaders
# ================================================================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/gt'),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
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
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# ================================================================
# Loops
# ================================================================
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')

# ================================================================
# Eval / logs / ckpts / vis
# ================================================================
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=log_interval),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend', save_dir=work_dir),
    dict(type='TensorboardVisBackend', save_dir=work_dir),
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
