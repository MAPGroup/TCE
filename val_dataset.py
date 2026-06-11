from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class dehaze_val_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.root_hazy = test_dir
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        try:
            files = os.listdir(test_dir)
        except Exception:
            files = []
        files = sorted(files)
        self.list_test = [f for f in files if (
            os.path.isfile(os.path.join(test_dir, f)) and os.path.splitext(f)[1].lower() in exts
        )]
        self.file_len = len(self.list_test)

    def __getitem__(self, index, is_train=True):
        path = os.path.join(self.root_hazy, self.list_test[index])
        hazy = Image.open(path).convert('RGB')
        hazy = self.transform(hazy)

#         hazy_up=hazy[:,0:1152,:]
#         hazy_down=hazy[:,48:1200,:]
#         print(hazy.shape)
        return hazy

    def __len__(self):
        return self.file_len
