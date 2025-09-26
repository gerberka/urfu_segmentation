# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../../configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py',
]

# ----------------------------------------------------------------
# Изменение гиперпараметров

# Поскольку мы используем только один графический процессор, вместо SyncBN используется BN
norm_cfg = dict(type='BN', requires_grad=True)
# Название датасета из файла urfu_project/dataset.py
dataset_type = 'RoadsDataset'
# Путь к папке с преобразованным набором данных
data_root = '/misc/home6/m_imm_freedata/Segmentation/landcover.ai_512'
# Количество классов для сегментации
num_classes = 2
# Размер изображения, который принимает на вход сеть
crop_size = (512, 512)
# Количество эпох для обучения
max_epochs = 100
# Функция потерь (переменная для удобства, но важно переопределить в decode_head)
loss = dict(type='FocalLoss', class_weight=[0.9, 1.1]) # [фон, таргет]
# Размер батча
batch_size = 16
gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps
# num_workers
num_workers = 8

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# Параметры логирования 
experiment_name = f'Mask2_{dataset_type}_{crop_size[0]}_{loss["type"]}_bsize_{actual_batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал в итерациях для печати логов

# Директория, где хрянятся файлы с списком изображений train и val
splits = 'splits'


pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa

depths = [2, 2, 18, 2]
# сохраняем backbone как есть, но добавляем переопределение head ниже
model = dict(
    backbone=dict(
        depths=depths, init_cfg=dict(type='Pretrained',
                                     checkpoint=pretrained))
)

model.update(
    dict(
        decode_head=dict(
            num_classes=num_classes,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                class_weight=[1.0, 3.0]  # фон=1.0, дорога=3.0 — подбирается эмпирически
            ),
            loss_mask=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
            loss_dice=dict(type='mmdet.DiceLoss', use_sigmoid=True, activate=True, eps=1.0, loss_weight=5.0),
        )
    )
)

# set all layers in backbone to lr_mult=0.1
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
# optimizer wrapper (включаем custom_keys)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0), clip_grad=None)

# ------------------ runtime / logging / scheduler ------------------
# Логгер и чекпойнты в стиле примера
log_processor = dict(by_epoch=True)

# простой полиномиальный шедулер (по эпохам) — можно сменить на любой другой
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=max_epochs,
        by_epoch=True)
]

# train loop — переопределяем, чтобы явно указать макс эпохи и валидацию каждую эпоху
train_cfg = dict(_delete_=True, max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)

# default_hooks: логирование, чекпоинты, визуализация (draw=False чтобы не сохранять много картинок автоматически)
default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=log_interval),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=3),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=500)
)

# ------------------ даталоадеры и пайплайны ------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
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

# ------------------ визуализация ------------------
vis_backends = [dict(type='LocalVisBackend', scalar_save_file='../../scalars.json', save_dir=work_dir),
                dict(type='TensorboardVisBackend', save_dir=work_dir)]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
