from wdcgan import Generator,same_seeds
from val_img_gen import generate_val_imgs
from pytorch_fid.fid_score import *
import glob
import shutil

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
    pth_path = glob.glob(r'wdcgan_g*.pth')[0]
    G = Generator(100)
    G.load_state_dict(torch.load(os.path.join(os.getcwd(), pth_path)))
    test(G)

if __name__ == '__main__':
    main()