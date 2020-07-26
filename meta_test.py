# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:01:15 2019

@author: WHX
"""


import torch
from model import SPCNNet
import meta_train
from matplotlib import pyplot as plt

def main():
    model = SPCNNet(3)
    model.load_state_dict(torch.load('meta_epochs/meta_trained_model'))
    meta_train.train_task(model,6);
    
    pic_num = [1,2,3,4,5]
    psnr_value1 = [19.605,17.416,19.919,23.034,21.782]
    psnr_value2 = [18.745,20.537,23.080,24.102,28.618]
    plt.plot(pic_num, psnr_value1, lw=2, ls='-', label="PSNR--Before Meta-Learning", color="r", marker="+")
    plt.plot(pic_num, psnr_value2, lw=2, ls='--', label="PSNR--After Meta-Learning", color="g", marker="+")
    plt.xlabel("training picture number/s", fontsize=16, horizontalalignment="center")
    plt.ylabel("PSNR value/db", fontsize=16, horizontalalignment="center")
    
    plt.legend()
    plt.savefig('D:\大三上\数字图像处理\SR_Project\plots\meta-learning_plot.png')
    plt.show()


if __name__ == '__main__':
    main()