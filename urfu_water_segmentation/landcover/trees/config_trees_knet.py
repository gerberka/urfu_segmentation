# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../../configs/knet/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py',
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
norm_cfg = dict(type='BN', requires_grad=True)
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
batch_size = 4
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


pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220308-f41b89d3.pth'  # noqa

depths = [2, 2, 18, 2]
num_stages = 3        # как в базовом конфиге
conv_kernel_size = 1  # как в базовом конфиге
num_classes = 2

model = dict(
    # оставляем Swin-бекбон, только глубины меняем
    backbone=dict(
        depths=depths,
    ),

    decode_head=dict(
        # 1) Полностью переопределяем kernel_update_head
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
            )
            for _ in range(num_stages)
        ],

        # 2) А здесь **дописываем** нужные поля, остальное тянем из базы
        kernel_generate_head=dict(
            num_classes=num_classes,
            # можешь оставить один CrossEntropyLoss, если не зайдёт MSE
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_weight=1.0)
            ],
        ),
    ),

    # 3) Auxiliary head тоже переводим на 2 класса
    auxiliary_head=dict(
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=0.4,
        ),
    ),
)

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    accumulative_counts=gradient_accumulation_steps,
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
)

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
