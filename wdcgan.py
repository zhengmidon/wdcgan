import  torch 
from    torch import nn, optim, autograd
import  numpy as np
#import  visdom
from    torch.nn import functional as F
#from    matplotlib import pyplot as plt
import  random
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
import shutil

class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames #a list storaging the names of pics
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
    '''
    '''
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        #resize the image to (64, 64)
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         # linearly map [0, 1] to [-1, 1],because generator use 'tanh' in the last layer    
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = FaceDataset(fnames, transform)
    return dataset


def same_seeds(seed):

	'''
	set seeds in order that it is possible to reproduce the training
	'''
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	np.random.seed(seed)  # Numpy module.
	random.seed(seed)  # Python random module.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class Generator(nn.Module):
    def __init__(self, z_dim, ):

        super().__init__()
        self.z_dim = z_dim
        net = []

        #   设定每次反卷积的输入和输出通道数等
        #   卷积核尺寸固定为4，反卷积输出为“SAME”模式
        channels_in = [self.z_dim, 512, 256, 128, 64]
        channels_out = [512, 256, 128, 64, 3]
        active = ["R", "R", "R", "R", "tanh"]
        stride = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]
        for i in range(len(channels_in)):
            net.append(nn.ConvTranspose2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                          kernel_size=4, stride=stride[i], padding=padding[i], bias=False))
            if active[i] == "R":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.ReLU())
            elif active[i] == "tanh":
                net.append(nn.Tanh())

        self.generator = nn.Sequential(*net)
        self.weight_init()
        
    def weight_init(self):
    	for m in self.generator.modules():
	        if isinstance(m, nn.ConvTranspose2d):
	            nn.init.normal_(m.weight.data, 0, 0.02)

	        elif isinstance(m, nn.BatchNorm2d):
	            nn.init.normal_(m.weight.data, 0, 0.02)
	            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        out = self.generator(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        """
        initialize
        
        :param image_size: tuple (3, h, w)
        """
        super().__init__()

        net = []
        # 1:预先定义
        channels_in = [3, 64, 128, 256, 512]
        channels_out = [64, 128, 256, 512, 1]
        padding = [1, 1, 1, 1, 0]
        active = ["LR", "LR", "LR", "LR", "sigmoid"]
        for i in range(len(channels_in)):
            net.append(nn.Conv2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                 kernel_size=4, stride=2, padding=padding[i], bias=False))
            if i == 0:
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "LR":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "sigmoid":
            	pass
                #net.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*net)
        self.weight_init()

    def weight_init(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        out = self.discriminator(x)
        out = out.view(-1)
        return out

def gradient_penalty(D, xr, xf, bs):
    """

    :param D:
    :param xr:
    :param xf:
    :return:
    """
    LAMBDA = 10

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(bs,1,1,1).cuda()

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp

from val_img_gen import generate_val_imgs
from pytorch_fid.fid_score import *

def validate(G,last_fid,retry):
    G.eval()
    device = torch.device('cuda')
    print(f'\rvalidating')

    temp_path = os.path.join(os.getcwd(),'Temp')
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

def main():

    # hyperparameters 
    batch_size = 256
    z_dim = 100
    lr = 7e-4
    #lr = 6e-4
    n_epoch = 80

    workspace_dir=os.getcwd()
    #the path to save pics generated in every epoch
    save_dir = os.path.join(workspace_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)

    # model
    G = Generator(z_dim = z_dim).cuda()
    D = Discriminator().cuda()
    G.train()
    D.train()

    # optimizer
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr,)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr,)


    same_seeds(42)
    # dataloader (You might need to edit the dataset path if you use extra dataset.)
    #dataset = get_dataset(os.path.join('/content','data', 'faces'))
    dataset = get_dataset(os.path.join(workspace_dir, 'crypko_data','faces'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # for logging
    z_sample = Variable(torch.randn(100, z_dim,1,1)).cuda()

    last_fid = 10000
    retry = 9
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim,1,1)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            # compute loss
            r_loss = -D(r_imgs.detach()).mean() #negtive because maximum
            f_loss = D(f_imgs.detach()).mean()
            #print(r_imgs.shape,f_imgs.shape)
            loss_D = r_loss + f_loss + gradient_penalty(D,r_imgs,f_imgs,bs)

            # update model
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            #train discriminator 5 times then train generator
            '''
            if (i+1) % 10 != 0 and i != len(dataloader)-1:
                continue
            '''

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim,1,1)).cuda()
            f_imgs = G(z)

            # compute loss
            loss_G = -D(f_imgs).mean() #negtive because maximum

            # update model
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            if (i + 1) % 100 == 0:
                print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}')

        G.eval()
        last_fid,save_flag,retry = validate(G,last_fid,retry)
        if save_flag:
            print(f'\rearly stopping at epoch {e+1},saving checkpoint')
            torch.save(G.state_dict(), os.path.join(workspace_dir, f'wdcgan_g_{(e+1)}.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, f'wdcgan_d_{(e+1)}.pth'))
            break
        with torch.no_grad():
            f_imgs_sample = (G(z_sample).data + 1) / 2.0 #project to [0,1]
            filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            print(f' | Save some samples to {filename}.')

        G.train()
        #set checkpoint
        if (e+1) % 25 == 0:
            torch.save(G.state_dict(), os.path.join(workspace_dir, f'wdcgan_g_{(e+1)}.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, f'wdcgan_d_{(e+1)}.pth'))


if __name__ == '__main__':
    main()
