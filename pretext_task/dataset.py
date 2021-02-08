import torchvision
from utils import *
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynDataset(Dataset):
    def __init__(self, cfg):
        super(SynDataset, self).__init__()
        self.indexlist = [line.rstrip('\n') for line in open(cfg.syn_img_list, 'r')]
        self.cfg = cfg
        self.norm = transforms.Normalize(self.cfg.mean, self.cfg.std)

    def __getitem__(self, idx):
        while True:
            info = self.indexlist[idx].split()
            if info[0] == 'suncg':
                color_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'mlt', info[1] + '_256.png'))
                depth_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'depth', info[1] + '_256.png'))
                edge_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'edge', info[1] + '_256.png'))
                normal_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'normal', info[1] + '_256.png'))
            elif info[0] == 'scenenet':
                color_img = io.imread(os.path.join(self.cfg.syn_path, info[1], 'photo', info[2] + '.jpg'))
                depth_img = io.imread(os.path.join(self.cfg.syn_path, info[1], 'depth', info[2] + '.png'))
                instance_img = io.imread(os.path.join(self.cfg.syn_path, info[1], 'instance', info[2] + '.png'))
                edge_img = io.imread(os.path.join(self.cfg.syn_path, info[1], 'edge', info[2] + '.png'))
                normal_img = io.imread(os.path.join(self.cfg.syn_path, info[1], 'normal', info[2] + '.png'))
            else:
                raise ValueError('wrong dataset!')

            depth_img[depth_img < 0] = 0
            # prevent
            depth_img = np.log(depth_img / 1000. + 1.0e-8) + 1.0e-8

            # choose the pics with decent edge map
            if np.count_nonzero(edge_img) < 350 or np.max(depth_img) > 3.4e+38 or np.min(depth_img) < -3.4e+38:
                idx = np.random.randint(len(self.indexlist))
            else:
                break

        # sample = {'color': color_img, 'depth': depth_img, 'edge': edge_img, 'normal': normal_img}
        sample = {'color': color_img, 'instance': instance_img, 'edge': edge_img, 'normal': normal_img}
        # print("ndarray_dataset", depth_img.sum(), np.max(depth_img))
        # image transformation
        _transforms = torchvision.transforms.Compose([Rescale(self.cfg.crop_size),
                                                      ToTensor()])

        sample = _transforms(sample)
        sample['edge'] = torch.where(sample['edge'] > 0., torch.tensor(1.0), torch.tensor(0.0))
        sample['normal'] = sample['normal'].div(255.0)
        edge_c = torch.sum(sample['edge'] > 0, dtype=torch.float32)
        sample['edge_c'] = edge_c
        # only normalize RGB image
        if self.norm:
            sample['color'] = sample['color'].div(255.0)
            sample['color'] = self.norm(sample['color'])
        return sample

    def __len__(self):
        return len(self.indexlist)


class RealDataset(Dataset):
    def __init__(self, cfg):
        super(RealDataset, self).__init__()
        self.image_list = [line.rstrip('\n') for line in open(cfg.real_img_list, 'r')]
        self.trans = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                         transforms.Resize([cfg.crop_size] * 2),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg.mean, cfg.std)])
        self.cfg = cfg

    def __getitem__(self, item):
        relative_path = self.image_list[item].split()[0]
        img_path = self.cfg.real_path + relative_path
        img = io.imread(img_path)
        if len(np.array(img).shape) == 2:
            img = np.expand_dims(np.array(img), 2).repeat(3, axis=2)
        img = self.trans(img)
        return img

    def __len__(self):
        return len(self.image_list)


class TestDataset(Dataset):
    def __init__(self, cfg, trans):
        self.cfg = cfg
        self.trans = trans
        self.img_name = [i for i in os.listdir(self.cfg.test_path)]
        self.img_list = [os.path.join(self.cfg.test_path, j) for j in self.img_name]

    def __getitem__(self, item):
        path = self.img_list[item]
        img = io.imread(path)
        if self.trans:
            img = self.trans(img)
        return self.img_name[item], img

    def __len__(self):
        return len(self.img_list)
