# ================================================================
# Plan
# ================================================================
# 1) Keep KNet + Swin-L backbone
# 2) Iter-based training loop (max_iters), AMP + grad accumulation
# 3) Loss: Focal + Dice (decode head), keep aux CE
# 4) Refactor: variables up top + consistent experiment naming
# ================================================================

# ================================================================
# Base
# ================================================================
_base_ = '../../../configs/knet/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py'

# ================================================================
# Dataset
# ================================================================
dataset_type = 'TreesDataset'
data_root = '/misc/home6/m_imm_freedata/Segmentation/Trees/Trees_DFC_512'

num_classes = 2
crop_size = (512, 512)

# Labels / ignore
ignore_index = 255
reduce_zero_label = False

custom_imports = dict(
    imports=[
        'transforms.debug_gt',
        'transforms.ensure_single_channel_gt',
        'transforms.sanitize_binary_gt',
    ],
    allow_failed_imports=False,
)

# ================================================================
# Training (high-level)
# ================================================================
max_iters = 80000
val_interval = 2000  # validate every N iters (reasonable default for iter-based loop)

batch_size = 4
num_workers = 8
log_interval = 10

# Hardware (for naming)
gpu_cnt = 8  # set your real GPU count if you want correct naming

# Grad accumulation
accumulative_counts = 2
effective_batch_size = batch_size * accumulative_counts
global_batch_size = effective_batch_size * gpu_cnt

# ================================================================
# Optimizer / schedule params
# ================================================================
optimizer_type = 'AdamW'
base_lr = 6e-5
weight_decay = 0.05
betas = (0.9, 0.999)

# Layer decay (Swin-L is deep; these are sane defaults; tune if needed)
layer_decay_num_layers = 24
layer_decay_rate = 0.90

# Warmup (iter-based)
warmup_iters = 1500
warmup_start_factor = 1e-3

# ================================================================
# Loss params
# ================================================================
loss_name = 'Focal_Dice'
focal_gamma = 2.0
focal_alpha = 0.25

# ================================================================
# Pretrain
# ================================================================
checkpoint_file = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/'
    'swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'
)

# ================================================================
# Experiment name
# ================================================================
experiment_name = (
    f'knet-s3'
    f'__bb=Swin-L'
    f'__ds=Trees_DFC_512'
    f'__classes={num_classes}'
    f'__crop={crop_size[0]}x{crop_size[1]}'
    f'__loss={loss_name}'
    f'__focal={focal_gamma:g}-{focal_alpha:g}'
    f'__opt={optimizer_type}'
    f'__lr={base_lr:g}'
    f'__wd={weight_decay:g}'
    f'__betas={betas[0]:g}-{betas[1]:g}'
    f'__acc={accumulative_counts}'
    f'__bs={batch_size}'
    f'__globbs={global_batch_size}'
    f'__ld={layer_decay_rate:g}'
    f'__ldepl={layer_decay_num_layers}'
    f'__warm={warmup_iters}it@{warmup_start_factor:g}'
    f'__sched=poly'
    f'__iters={max_iters}'
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
    pretrained=None,
    data_preprocessor=data_preprocessor,

    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.4,  # try: 0.2 / 0.3 / 0.4
        patch_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    ),

    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=3,

        kernel_generate_head=dict(
            type='UPerHead',
            ignore_index=ignore_index,
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            out_channels=1,
            threshold=0.5,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=focal_gamma,
                    alpha=focal_alpha,
                    loss_weight=1.0,
                ),
                dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    loss_weight=1.0,
                    ignore_index=ignore_index,
                ),
            ],
        ),

        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=num_classes,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'),
                ),
            ),
            dict(
                type='KernelUpdateHead',
                num_classes=num_classes,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'),
                ),
            ),
            dict(
                type='KernelUpdateHead',
                num_classes=num_classes,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'),
                ),
            ),
        ],
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        out_channels=1,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.4,
        ),
    ),
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
    ),
    accumulative_counts=accumulative_counts,
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# Iter-based schedule (warmup + poly)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=warmup_start_factor,
        by_epoch=False,
        begin=0,
        end=warmup_iters,
    ),
    dict(
        type='PolyLR',
        power=1.0,
        by_epoch=False,
        begin=warmup_iters,
        end=max_iters,
        eta_min=0.0,
    ),
]

# ================================================================
# Pipelines
# ================================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),

    dict(type='EnsureSingleChannelGT'),
    dict(type='SanitizeBinaryGT'),

    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),

    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),

    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),

    dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
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
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# ================================================================
# Loops (ITER-BASED)
# ================================================================
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval,
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
        by_epoch=False,          # important for iter-based training
        interval=val_interval,   # checkpoint on same cadence as val (change if you want)
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