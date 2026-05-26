# split_dataset.py
import os, shutil, random

random.seed(42)

src_images = 'dataset/train/images'
src_labels = 'dataset/train/labels'

for split in ['valid', 'test']:
    os.makedirs(f'dataset/{split}/images', exist_ok=True)
    os.makedirs(f'dataset/{split}/labels', exist_ok=True)

images = os.listdir(src_images)
random.shuffle(images)

n = len(images)
valid_imgs = images[:int(n * 0.15)]
test_imgs  = images[int(n * 0.15):int(n * 0.30)]

for img in valid_imgs:
    shutil.move(f'{src_images}/{img}', f'dataset/valid/images/{img}')
    label = img.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(f'{src_labels}/{label}'):
        shutil.move(f'{src_labels}/{label}', f'dataset/valid/labels/{label}')

for img in test_imgs:
    shutil.move(f'{src_images}/{img}', f'dataset/test/images/{img}')
    label = img.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(f'{src_labels}/{label}'):
        shutil.move(f'{src_labels}/{label}', f'dataset/test/labels/{label}')

print(f"Total: {n} | Train: {n - int(n*0.30)} | Valid: {len(valid_imgs)} | Test: {len(test_imgs)}")