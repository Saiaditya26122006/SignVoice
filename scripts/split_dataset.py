import os
import shutil
import random

source = 'data/isl_dataset'
output = 'data/isl_dataset_split'
random.seed(42)

for class_name in sorted(os.listdir(source)):
    class_path = os.path.join(source, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    n = len(images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    for split, files in splits.items():
        split_dir = os.path.join(output, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(class_path, f), os.path.join(split_dir, f))
    print(f'Done: {class_name}')

print('Split complete')
