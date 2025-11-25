from collections import defaultdict
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


DATASET_COLORMAP = dict(
    background=(0, 0, 0),
    water=(64, 64, 64),
    tree=(128, 128, 128),
    roads=(192, 192, 192),
    buildings=(255, 255, 255),
    clouds=(210, 210, 210),
)

WATER_CLASS_MAPPING = {
    'tree': 'background',
    'roads': 'background',
    'buildings': 'background',
    'clouds': 'background'
}

TREES_CLASS_MAPPING = {
    'water': 'background',
    'roads': 'background',
    'buildings': 'background',
    'clouds': 'background',
}

CLOUDS_CLASS_MAPPING = {
    'tree': 'background',
    'roads': 'background',
    'buildings': 'background',
    'water': 'background',
}

ROADS_CLASS_MAPPING = {
    'water': 'background',
    'buildings': 'background',
    'clouds': 'background',
    'tree': 'background',
}


@DATASETS.register_module()
class RoadsDataset(BaseSegDataset):
    METAINFO = dict(
        classes=list(DATASET_COLORMAP.keys()),
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        new_classes = []
        self._data_label_map = {}

        for key, value in DATASET_COLORMAP.items():
            new_key = ROADS_CLASS_MAPPING.get(key, key)
            if new_key not in new_classes:
                new_classes.append(new_key)
            self._data_label_map[value[0]] = new_classes.index(new_key)

        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            metainfo={'classes': new_classes},
            **kwargs
        )

    def get_data_info(self, idx):
        result = super().get_data_info(idx)
        if result is None:
            return None

        result['label_map'] = self._data_label_map.copy()
        return result

@DATASETS.register_module()
class WaterDataset(BaseSegDataset):
    METAINFO = dict(
        classes=list(DATASET_COLORMAP.keys()),
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        new_classes = []
        self._data_label_map = {}
        
        for key, value in DATASET_COLORMAP.items():
            new_key = WATER_CLASS_MAPPING.get(key, key)
            
            if new_key not in new_classes:
                new_classes.append(new_key)
            
            self._data_label_map[value[0]] = new_classes.index(new_key)
            
        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            metainfo={'classes': new_classes},
            **kwargs
        )
        
    def get_data_info(self, idx):
        result = super().get_data_info(idx)
        if result is None:
            return None
        
        result['label_map'] = self._data_label_map.copy()
        return result


@DATASETS.register_module()
class TreesDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'tree'),
        palette=[(0, 0, 0), (128, 128, 128)],
    )

    def __init__(self, **kwargs):
        # Мапим любое значение пикселя в маске в {0,1} чтобы не ловить
        # `index out of bounds` в cuda: все -> фон, только 128 -> дерево.
        self._data_label_map = {i: 0 for i in range(256)}
        self._data_label_map[128] = 1

        super().__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            # важно: не трогаем zero-label
            reduce_zero_label=False,
            # ignore_index можешь оставить 255, но мы 255 уже мапим в 0
            **kwargs,
        )

    def get_data_info(self, idx):
        result = super().get_data_info(idx)
        if result is None:
            return None

        # mmseg возьмет отсюда label_map и применит его в LoadAnnotations
        result['label_map'] = self._data_label_map.copy()
        return result

    def load_data_list(self):
        """Проставляем label_map заранее, чтобы рабочие процессы точно
        получали маппинг масок в {0,1} при загрузке аннотаций."""
        data_list = super().load_data_list()
        for item in data_list:
            item['label_map'] = self._data_label_map.copy()
        return data_list


@DATASETS.register_module()
class CloudsDataset(BaseSegDataset):
    METAINFO = dict(
        classes=list(DATASET_COLORMAP.keys()),
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        new_classes = []
        self._data_label_map = {}
        
        for key, value in DATASET_COLORMAP.items():
            new_key = CLOUDS_CLASS_MAPPING.get(key, key)
            if new_key not in new_classes:
                new_classes.append(new_key)
            self._data_label_map[value[0]] = new_classes.index(new_key)
        
        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            metainfo={'classes': new_classes},
            **kwargs
        )
        
    def get_data_info(self, idx):
        result = super().get_data_info(idx)
        if result is None:
            return None
        result['label_map'] = self._data_label_map.copy()
        return result    
