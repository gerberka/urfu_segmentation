_base_ = '../../../configs/knet/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py'

# ----------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------
dataset_type = 'TreesDataset'
data_root = '/misc/home6/m_imm_freedata/Segmentation/Trees/Trees_DFC_512'

num_classes = 2
crop_size = (512, 512)

metainfo = dict(
    classes=('background', 'tree'),
    palette=[(0, 0, 0), (0, 255, 0)],
)

# ----------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------
max_epochs = 200
batch_size = 4
num_workers = 8

experiment_name = (
    f'KNet_SwinL_TreesDS_CE_{num_classes}cls_{crop_size[0]}crop_AdamW_{max_epochs}ep'
)
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'
log_interval = 10

# ----------------------------------------------------------------
# PRETRAIN
# ----------------------------------------------------------------
checkpoint_file = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/'
    'swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'
)

# ----------------------------------------------------------------
# PREPROCESSOR
# ----------------------------------------------------------------
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255,
)

# ----------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------
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
        drop_path_rate=0.4,
        patch_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    ),

    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=3,
        kernel_generate_head=dict(
            type='UPerHead',
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
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
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'),
                    act_cfg=None,
                ),
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
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'),
                    act_cfg=None,
                ),
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
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'),
                    act_cfg=None,
                ),
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
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
)

# ----------------------------------------------------------------
# PIPELINES
# ----------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
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
    dict(type='PackSegInputs'),
]

# ----------------------------------------------------------------
# DATALOADERS
# ----------------------------------------------------------------
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/gt',
        ),
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
        metainfo=metainfo,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/gt',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# ----------------------------------------------------------------
# LOOPS
# ----------------------------------------------------------------
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')

# ----------------------------------------------------------------
# EVALUATOR / VISUALIZER
# ----------------------------------------------------------------
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=log_interval),
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
