from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import shutil
#from val_img_gen import generate_val_imgs
from pytorch_fid.fid_score import *


class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = FaceDataset(fnames, transform)
    return dataset

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())
        self.apply(weights_init)        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

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

def validate(G,last_fid,retry):
    G.eval()
    device = torch.device('cuda')
    print(f'\rvalidating')

    temp_path = os.path.join(os.getcwd(),'Temp_base')
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        os.mkdir(temp_path)
    else:
        os.mkdir(temp_path)

    #generating images for validation
    generate_val_imgs(G,temp_path,device,nimgs = 2000)

    val_paths = [temp_path,os.path.join(os.getcwd(),'test')]
    #employ pytorch_fid to calculate fid score
    fid_value = calculate_fid_given_paths(val_paths,
                                          50,
                                          device,
                                          2048,
                                          8)
    fid_value = round(fid_value,2)
    #early stop
    if fid_value <= last_fid :
        retry = 9
        last_fid = fid_value
        save_chk = False
    elif fid_value > last_fid and retry > 0 :
        retry -=1
        save_chk = False
    else:
        save_chk = True

    print(f'\rfid score:{last_fid}')
    shutil.rmtree(temp_path)

    return (last_fid,save_chk,retry)


# hyperparameters 
batch_size = 128
z_dim = 100
lr = 2e-4
n_epoch = 50
workspace_dir = os.getcwd()
save_dir = os.path.join(workspace_dir, 'logs_base')
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda")

# model
G = Generator(in_dim=z_dim).to(device)
D = Discriminator(3).to(device)
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


same_seeds(42)
# dataloader (You might need to edit the dataset path if you use extra dataset.)
dataset = get_dataset(os.path.join(workspace_dir, 'crypko_data','faces'))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# for logging
z_sample = Variable(torch.randn(100, z_dim)).to(device)

last_fid = 10000
retry = 9
for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.to(device)

        bs = imgs.size(0)

        """ Train D """
        z = Variable(torch.randn(bs, z_dim)).to(device)
        r_imgs = Variable(imgs).to(device)
        f_imgs = G(z)

        # label        
        r_label = torch.ones((bs)).to(device)
        f_label = torch.zeros((bs)).to(device)

        # dis
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        
        # compute loss
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = (r_loss + f_loss) / 2

        # update model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        """ train G """
        # leaf
        z = Variable(torch.randn(bs, z_dim)).to(device)
        f_imgs = G(z)

        # dis
        f_logit = D(f_imgs)
        
        # compute loss
        loss_G = criterion(f_logit, r_label)

        # update model
        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # log
        if (i + 1) % 100 == 0:
            print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}')
    G.eval()
    # validation
    last_fid,save_flag,retry = validate(G,last_fid,retry)
    if save_flag:
        print(f'\rearly stopping at epoch {e+1},saving checkpoint')
        torch.save(G.state_dict(), os.path.join(workspace_dir, f'dcgan_g_base_{(e+1)}.pth'))
        torch.save(D.state_dict(), os.path.join(workspace_dir, f'dcgan_d_base_{(e+1)}.pth'))
        break

    #record the intermediate results
    with torch.no_grad():
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')
    G.train()
    #save checkpoint
    if (e+1) % 25 == 0:
        torch.save(G.state_dict(), os.path.join(workspace_dir, f'dcgan_g_base_{e+1}.pth'))
        torch.save(D.state_dict(), os.path.join(workspace_dir, f'dcgan_d_base_{e+1}.pth'))
