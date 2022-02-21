import torch
import os
#from wdcgan import *
from torchvision import transforms
from torch.autograd import Variable

def generate_val_imgs(G,save_path,device,nimgs = 2000):
    z_dim = 100
    workspace_dir=os.getcwd()

    G.eval()
    G.to(device)

    # generate images and save the result

    z_sample = Variable(torch.randn(nimgs, z_dim,1,1)).to(device)
    with torch.no_grad():
        imgs_sample = (G(z_sample).data + 1) / 2.0

    save_dir = os.path.join(workspace_dir, save_path)
    toPIL = transforms.ToPILImage()
    for i,img in enumerate(imgs_sample):
        filename = os.path.join(save_dir, f'gen_{i}.jpg')
        pic = toPIL(img)
        pic.save(filename)

def main():
    G = Generator(z_dim)
    G.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))
    generate_val_imgs()

if __name__ == '__main__':
    main()