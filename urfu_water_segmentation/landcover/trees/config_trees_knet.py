# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../../configs/knet/knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512.py',
]

default_scope = 'mmseg'

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, start_factor=1e-3),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5, end=100, T_max=95),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1,
                    save_best='mIoU', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10)
)


# ----------------------------------------------------------------
# Изменение гиперпараметров

# Если используем один графический процессор - BN, иначе - SyncBN
norm_cfg = dict(type='SyncBN', requires_grad=True)
# Название датасета из файла urfu_project/dataset.py
dataset_type = 'TreesDataset'
# Путь к папке с преобразованным набором данных
data_root = '/misc/home6/m_imm_freedata/Segmentation/Trees/Trees_DFC_512'
# Количество классов для сегментации
num_classes = 2
num_stages = 3
# Размер изображения, который принимает на вход сеть
crop_size = (512, 512)
# Количичество эпох для обучения
max_epochs = 100
# Функция потерь
loss = [
        dict(type='CrossEntropyLoss', loss_weight=1.0),
    ]
# Размер батча
batch_size = 16
gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps
# num_workers
num_workers = 8

# Оптимизатор
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.001)

# Параметры логирования 
experiment_name = f'KNET_CEMSE_{dataset_type}_{crop_size[0]}_' + '_'.join([l['type'] for l in loss]) + f'_bsize_{actual_batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал в итерациях для печати логов

# Директория, где хрянятся файлы с списком изображений train и val
splits = 'splits'

depths = [2, 2, 18, 2]
num_stages = 3        # как в базовом конфиге
conv_kernel_size = 1  # как в базовом конфиге
num_classes = 2

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=num_stages,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=num_classes,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(num_stages)
        ],
        kernel_generate_head=dict(
            type='UPerHead',
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))),
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
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0005),
    clip_grad=dict(max_norm=1, norm_type=2))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/gt'),
        pipeline=train_pipeline,
        # ann_file=f'{splits}/train.txt'
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
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/gt'),
        pipeline=test_pipeline,
        # ann_file=f'{splits}/val.txt'
        )
    )

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend', scalar_save_file='scalars.json', save_dir=work_dir),
    dict(type='TensorboardVisBackend', save_dir=work_dir),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
