import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_generate import is_image_file
from model import SPCNNet

def main(factor):
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epoch_3_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    if factor != 3:
        UPSCALE_FACTOR = factor
    MODEL_NAME = opt.model_name
    if MODEL_NAME[6] != str(UPSCALE_FACTOR):
        MODEL_NAME = 'epoch_' + str(UPSCALE_FACTOR) + opt.model_name[-7:-1] + 't'
        

    path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/'
    images_name = [x for x in listdir(path) if is_image_file(x)]
    model = SPCNNet(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:                                           
        with tqdm(images_name, desc='converting LR images to HR images') as ttt:
            for image_name in ttt:

                img = Image.open(path + image_name).convert('YCbCr')
                y, cb, cr = img.split()
                image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
                if torch.cuda.is_available():
                    image = image.cuda()

                out = model(image)
                out = out.cpu()
                out_img_y = out.data[0].numpy()
                out_img_y *= 255.0
                out_img_y = out_img_y.clip(0, 255)
                out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
                out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
                out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
                out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
                out_img.save(out_path + image_name)
    except KeyboardInterrupt:
        ttt.close()
        raise
    ttt.close()

if __name__ == "__main__":
    main(3)


