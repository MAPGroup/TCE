from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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

class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.pairs = []
        self.root_haze, self.root_clean = find_pair_roots(test_dir)
        clean_index = build_clean_index(self.root_clean)
        txt_path = os.path.join(test_dir, 'test.txt')
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

    def __getitem__(self, index, is_train=True):
        file_name, clean_name = self.pairs[index]
        haze = Image.open(os.path.join(self.root_haze, file_name)).convert('RGB')
        clean = Image.open(os.path.join(self.root_clean, clean_name)).convert('RGB')
        haze = self.transform(haze)

        haze_up=haze[:,0:640,:]
        haze_down=haze[:,560:1200,:]
        clean = self.transform(clean)
        return file_name, haze_up, haze_down, haze, clean

    def __len__(self):
        return self.file_len
