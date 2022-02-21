#from dcgan import Generator,same_seeds
#from val_img_gen import generate_val_imgs
import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_fid.fid_score import *
import glob
import shutil
import random
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms

class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generate_val_imgs(G,save_path,device,nimgs = 2000):
    z_dim = 100
    workspace_dir=os.getcwd()

    G.eval()
    G.to(device)

    # generate images and save the result

    z_sample = Variable(torch.randn(nimgs, z_dim)).to(device)
    with torch.no_grad():
        imgs_sample = (G(z_sample).data + 1) / 2.0

    save_dir = os.path.join(workspace_dir, save_path)
    toPIL = transforms.ToPILImage()
    for i,img in enumerate(imgs_sample):
        filename = os.path.join(save_dir, f'gen_{i}.jpg')
        pic = toPIL(img)
        pic.save(filename)

def test(G):
    G.eval()
    device = torch.device('cuda')
    print(f'\rtesting')

    temp_path = os.path.join(os.getcwd(),'Temp_test')
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        os.mkdir(temp_path)
    else:
        os.mkdir(temp_path)

    #generating images for test
    generate_val_imgs(G,temp_path,device,nimgs = 2000)

    val_paths = [temp_path,os.path.join(os.getcwd(),'test')]
    #employ pytorch_fid to calculate fid score
    fid_value = calculate_fid_given_paths(val_paths,
                                          50,
                                          device,
                                          2048,
                                          8)
    fid_value = round(fid_value,2)
    shutil.rmtree(temp_path)
    print(f'fid score is {fid_value}')

def main():
    same_seeds(42)
    pth_path = glob.glob(r'dcgan_g*.pth')[0]
    G = Generator(100)
    G.load_state_dict(torch.load(os.path.join(os.getcwd(), pth_path)))
    test(G)

if __name__ == '__main__':
    main()