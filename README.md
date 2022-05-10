## WDCGAN
Recording my learning process of GAN.
Baseline model is based on [this](https://colab.research.google.com/drive/1JYY_HHtVSSOLixZfLwkxiWTRdPHJCS2t).
For details please refer to [this paper](https://github.com/zhengmidon/wdcgan/blob/master/%E7%94%A8%E4%BA%8E%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F%E7%94%9F%E6%88%90%E7%9A%84%E6%94%B9%E8%BF%9B%20DCGAN.pdf)
### Directory Annotation
```
/
	|-- base_training.log 		#training log of baseline model
	|-- training.log 		#training log of improved model
	|-- dcgan.py 			#baseline model script
	|-- wdcgan.py 			#improved model script
	|-- test.py 			#test script of improved model
	|-- test_base.py 		#test script of baseline model
	|-- val_img_gen.py 		#some functional functions
	|-- *.pth 			#checkpoint files
	|-- crypko_data/ 		#put training data here 
	|-- logs/ 		    	#log pics of every epoch for improved model
	|-- logs_base/ 			# log pics of every epoch for baseline model
	|-- test/ 			# test set
```
### Environment Preparation
```
torch
torchvision
cv2
pytorch_fid
```
### Data Preparation

 Download training data from [BaiduNetdisk](https://pan.baidu.com/s/14Go5HFc0oZHut9CZoUe67Q)(extracting code:inle),uncompress it to **crypko_data/**
### Training
Make sure your machine has Nvidia GPUs,simply run
```bash
python wdcgan.py
```
for improved model,or
```
python dcgan.py
```
for baseline model.
### Test
```bash
python test.py
```
outputs **FID** score of improved model.
```bash
python test_base.py
```
outputs **FID** score of baseline model.

Sample image of improved model:

![Sample image](https://github.com/zhengmidon/wdcgan/blob/main/Epoch_060.jpg)


