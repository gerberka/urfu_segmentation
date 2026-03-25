# ================================================================
# Base
# ================================================================
_base_ = '../../../configs/san/san-vit-l14_coco-stuff164k-640x640.py'

# ================================================================
# Dataset
# ================================================================
dataset_type = 'TreesDataset'
data_root = '/misc/home6/m_imm_freedata/Segmentation/Trees/Trees_DFC_512'

num_classes = 2
crop_size = (512, 512)

custom_imports = dict(
    imports=[
        'transforms.ensure_single_channel_gt',
        'transforms.sanitize_binary_gt',
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

accumulative_counts = 2
effective_batch_size = batch_size * accumulative_counts

# Optimizer
optimizer_type = 'AdamW'
base_lr = 2e-5
weight_decay = 0.05
betas = (0.9, 0.999)

# Warmup
warmup_epochs = 5
warmup_start_factor = 1e-3

# Loss / ignore
ignore_index = 255
loss_name = 'CE'

# Slide inference
test_mode = 'slide'
slide_stride = (426, 426)

# Pretrain (тот же CLIP ViT-L/14)
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth'

# ================================================================
# Experiment name
# ================================================================
experiment_name = (
    f'san-vit-l14'
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
# mean/std будут перезаписаны из mean_vals.txt в train.py
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
model = dict(
    _delete_=True,
    type='MultimodalEncoderDecoder',
    pretrained=pretrained,
    data_preprocessor=data_preprocessor,
    encoder_resolution=0.7,
    image_encoder=dict(
        type='VisionTransformer',
        img_size=(336, 336),
        patch_size=14,
        patch_pad=0,
        embed_dims=1024,
        num_layers=18,
        num_heads=16,
        out_indices=(5, 11, 17),
    ),
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        output_dims=768,
        vocabulary=['background', 'tree'],
    ),
    decode_head=dict(
        type='SideAdapterCLIPHead',
        num_classes=num_classes,
        ignore_index=ignore_index,
        san_cfg=dict(clip_channels=1024, cfg_decoder=dict(num_heads=16)),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
        ),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
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
    accumulative_counts=accumulative_counts,
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
    _delete_=True,
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
    _delete_=True,
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
# Loops  (base использует IterBasedTrainLoop — переключаем на Epoch)
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