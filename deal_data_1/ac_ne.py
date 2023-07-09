# -*- coding: utf-8 -*-
# @Time : 2023/4/7 16:50
# @Author : longchen
# @File : ac_ne.py
# @Project : voxnet_M16_test


import numpy as np
import pandas as pd
import os, shutil,random,math
import matplotlib.pyplot as plt
import astropy.io.fits as fits

#得到正负样本

def get_ac(detect_sample_path, verify_yes, save_acpath):
    '''
    得到正样本
    :param detect_sample_path:
    :param verify_yes:
    :param save_acpath:
    :return:
    '''
    df_yes = pd.read_csv(verify_yes, sep='\t')
    core_list = df_yes['ID']  # 正样本
    for core_name in core_list:
        old_path = os.path.join(detect_sample_path, core_name + '.fits')
        if os.path.exists(old_path):
            new_path = os.path.join(save_acpath, core_name + '.fits')
            if not os.path.exists(new_path):
                shutil.copy(old_path, new_path)

# def get_ne(len):


#改名字和移动位置
def get_labelcore(origin_name,label,des_dir):
    '''
    
    :param origin_name: 
    :param label: 
    :param des_dir: 
    :return: 
    '''
    new_name=label+"_"+origin_name.split('/')[-1]
    new_path=os.path.join(des_dir,new_name)
    shutil.copy(origin_name,new_path)

if __name__ == '__main__':
    #R2 5rms
    #正样本
    # detect_sample_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/R2_200_L'
    # cell_list=[i for i in os.listdir(detect_sample_dir)]
    # verify_yes = r'/home/data/clumps_share/data_longc/云核认证/R2_LDC/first_verify/Yes_LDC_R2.csv'
    # save_acpath = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/ac_sample_1'
    # for i, item in enumerate(cell_list):
    #     detect_sample_path=os.path.join(detect_sample_dir,item)
    #     get_ac(detect_sample_path, verify_yes, save_acpath)
    #R2不是云核的
    # detect_sample_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/R2_200_L'
    # cell_list=[i for i in os.listdir(detect_sample_dir)]
    # verify_yes = r'/home/data/clumps_share/data_longc/云核认证/R2_LDC/second_verify/No_LDC_R2.csv'
    # save_acpath = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/no_clump_ldc'
    # os.makedirs(save_acpath,exist_ok=True)
    # for i, item in enumerate(cell_list):
    #     detect_sample_path=os.path.join(detect_sample_dir,item)
    #     get_ac(detect_sample_path, verify_yes, save_acpath)



    #负样本
    # false_path=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/False_clump'
    # cell_list=os.listdir(false_path)
    # ac_sample=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/ac_sample'
    # ac_sum=len(os.listdir(ac_sample))
    # print(ac_sum)
    # ne_sample=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/ne_sample'
    # shutil.rmtree(ne_sample)
    # os.mkdir(ne_sample)
    # ne_sum=0
    # for i,item in enumerate(cell_list):
    #     cell_path=os.path.join(false_path,item)
    #     fits_list=os.listdir(cell_path)
    #     cell_len = len(fits_list)
    #     ne_sum+=cell_len
    # print(ne_sum)
    #
    # for i,item in enumerate(cell_list):
    #     cell_path=os.path.join(false_path,item)
    #     fits_list=os.listdir(cell_path)
    #     cell_len=len(fits_list)
    #     num_sample=math.ceil(ac_sum*(cell_len/ne_sum))
    #     sample_list=random.sample(fits_list,num_sample)#每个cell里面取16307*（cell/all_false）个假团块
    #     for j,fits_data in enumerate(sample_list):
    #         fits_path=os.path.join(cell_path,fits_data)
    #         new_path=os.path.join(ne_sample,fits_data)
    #         shutil.copy(fits_path,new_path)


    #创造训练集
    train_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/train_data_1'
    os.makedirs(train_dir,exist_ok=True)
    # test_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/test_data_1'
    # os.makedirs(test_dir, exist_ok=True)
    ac_sample=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/ac_sample'
    ac_list=os.listdir(ac_sample)
    ac_train=random.sample(ac_list,7440)
    # ac_train=[i for i in ac_list if int(i[7])<8]
    # ac_test=[i for i in ac_list if int(i[7])>=8]
    # print('正样本训练集样本个数：%d,测试集个数：%d'%(len(ac_train),len(ac_test)))
    for i,item in enumerate(ac_train):
        origin_ac=os.path.join(ac_sample,item)
        get_labelcore(origin_ac, '1', train_dir)  #正样本1开头
    # for i, item in enumerate(ac_test):
    #     origin_ac = os.path.join(ac_sample, item)
    #     get_labelcore(origin_ac, '1', test_dir)  #正样本1开头

    ne_sample = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/noclump_ldc_fits'
    ne_list = os.listdir(ne_sample)
    # ne_train = [i for i in ne_list if int(i[7]) < 8]
    # ne_test = [i for i in ne_list if int(i[7]) >= 8]
    # print('负样本训练集样本个数：%d,测试集个数：%d' % (len(ne_train), len(ne_test)))
    for i,item in enumerate(ne_list):
        origin_ne=os.path.join(ne_sample,item)
        get_labelcore(origin_ne, '0', train_dir)    #负样本0开头
    # for i,item in enumerate(ne_test):
    #     origin_ne=os.path.join(ne_sample,item)
    #     get_labelcore(origin_ne, '0', test_dir)    #负样本0开头