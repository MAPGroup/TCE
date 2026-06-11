from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
import random
import os

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
PAIR_DIR_NAMES = [
    ('haze', 'clear'),
    ('hazy', 'GT'),
    ('hazy', 'gt'),
    ('hazy', 'clear'),
]


def find_pair_roots(root_dir):
    for haze_name, clean_name in PAIR_DIR_NAMES:
        root_haze = os.path.join(root_dir, haze_name)
        root_clean = os.path.join(root_dir, clean_name)
        if os.path.isdir(root_haze) and os.path.isdir(root_clean):
            return root_haze, root_clean
    raise FileNotFoundError(
        'Cannot find paired image folders under {}. Expected one of: {}'.format(
            root_dir,
            ', '.join('{}/{}'.format(h, c) for h, c in PAIR_DIR_NAMES)
        )
    )


def build_clean_index(root_clean):
    index = {}
    for clean_name in os.listdir(root_clean):
        clean_path = os.path.join(root_clean, clean_name)
        if not os.path.isfile(clean_path):
            continue
        clean_stem, clean_ext = os.path.splitext(clean_name)
        if clean_ext.lower() not in IMG_EXTS:
            continue
        index[clean_name.lower()] = clean_name
        index[clean_stem.lower()] = clean_name
    return index


def find_clean_name(haze_name, clean_index):
    haze_stem, haze_ext = os.path.splitext(haze_name)
    candidates = [
        haze_name.lower(),
        haze_stem.lower(),
        (haze_stem.split('_')[0] + haze_ext).lower(),
        haze_stem.split('_')[0].lower(),
    ]
    for candidate in candidates:
        if candidate in clean_index:
            return clean_index[candidate]
    return None

#data augmentation for image rotate
def augment(haze, clean):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        haze = transforms.functional.rotate(haze, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        return haze, clean
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        haze = vertical_flip(haze)
        clean = vertical_flip(clean)
        return haze, clean
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        haze = horizontal_flip(haze)
        clean = horizontal_flip(clean)
        return haze, clean
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return haze, clean


class dehaze_train_dataset(Dataset):
    def __init__(self, train_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.pairs = []
        self.root_haze, self.root_clean = find_pair_roots(train_dir)
        clean_index = build_clean_index(self.root_clean)
        txt_path = os.path.join(train_dir, 'train.txt')
        if os.path.exists(txt_path):
            for line in open(txt_path):
                line = line.strip()
                if line != '':
                    clean_name = find_clean_name(line, clean_index)
                    if clean_name is not None:
                        self.pairs.append((line, clean_name))
        else:
            try:
                files = os.listdir(self.root_haze)
            except Exception:
                files = []
            files = sorted(files)
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in IMG_EXTS:
                    haze_path = os.path.join(self.root_haze, f)
                    clean_name = find_clean_name(f, clean_index)
                    clear_path = os.path.join(self.root_clean, clean_name) if clean_name else ''
                    if os.path.isfile(haze_path) and os.path.isfile(clear_path):
                        self.pairs.append((f, clean_name))
        self.file_len = len(self.pairs)

    def __getitem__(self, index, is_train = True):
        if is_train:
            haze_name, clean_name = self.pairs[index]
            haze = Image.open(os.path.join(self.root_haze, haze_name)).convert('RGB')
            clean = Image.open(os.path.join(self.root_clean, clean_name)).convert('RGB')
            #crop a patch
            i,j,h,w = transforms.RandomCrop.get_params(haze, output_size = (256,256))
            haze_ = TF.crop(haze, i, j, h, w)
            clean_ = TF.crop(clean, i, j, h, w)

            #data argumentation
            haze_arg, clean_arg = augment(haze_, clean_)
        haze = self.transform(haze_arg)
        clean = self.transform(clean_arg)
        return haze,clean

    def __len__(self):
        return self.file_len
