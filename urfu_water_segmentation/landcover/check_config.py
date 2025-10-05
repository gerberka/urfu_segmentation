# -*- coding: utf-8 -*-
import argparse
from mmengine.config import Config
import json

parser = argparse.ArgumentParser(description="Save config to JSON or TXT")
parser.add_argument("--txt", action="store_true", help="Save as .txt instead of .json")
args = parser.parse_args()


cfg = Config.fromfile('roads_config/config_landcover_mask2_baseline.py')
#config_to_save = cfg.model.get('decode_head', cfg.model)
config_to_save = cfg.model

output_file = "config_output.txt" if args.txt else "config_output.json"
with open(output_file, 'w', encoding='utf-8') as f:
    if args.txt:
        f.write(str(config_to_save))
    else:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)

print(f"Config saved in {output_file}")
