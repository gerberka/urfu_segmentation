import os

path = '/misc/home6/m_imm_freedata/Segmentation/landcover.ai_512/train/gt2'
file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

print(f"Количество файлов в папке: {file_count}")
