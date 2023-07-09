# -*- coding: utf-8 -*-
# @Time : 2023/4/7 15:05
# @Author : longchen
# @File : get_clumps.py
# @Project : voxnet_M16_test


import numpy as np
import pandas as pd
import os, math,random,shutil
import astropy.io.fits as fits
from multiprocessing import Pool


def get_clumps(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir):
    """
    将每个云核切出来，放入立方体数据块中，保存为fits文件到save_dir文件夹
    :param mask_name: 云核检测掩模
    :param loc_outcat_name: 局部像素核表
    :param loc_outcat_wcs_name: 局部wcs核表
    :param origin_fit:原始云核数据
    :param save_dir: 切出来的小立方体保存的文件夹
    :return:
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    mask_data = fits.getdata(mask_name)  # 每个cell的mask
    outcat_loc = pd.read_csv(loc_outcat_name, sep='\t')
    outcat_loc_wcs = pd.read_csv(loc_outcat_wcs_name, sep='\t')
    origin_data = fits.getdata(origin_fit)

    for i in range(outcat_loc.shape[0]):
        clump_i = outcat_loc.iloc[i]
        clump_i_wcs = outcat_loc_wcs.iloc[i]
        clump_id_wcs = clump_i_wcs['ID']
        clump_id = clump_i['ID']

        core_x = np.array(np.where(mask_data == clump_id)[0])
        core_y = np.array(np.where(mask_data == clump_id)[1])
        core_z = np.array(np.where(mask_data == clump_id)[2])
        x_min = core_x.min()
        x_max = core_x.max()
        y_min = core_y.min()
        y_max = core_y.max()
        z_min = core_z.min()
        z_max = core_z.max()

        mask_copy = mask_data.copy()
        mask_copy[mask_copy != clump_id] = 0
        mask_copy = mask_copy / clump_id

        multiply_data = origin_data * mask_copy  # mask与原始数据相乘
        core_data = multiply_data[x_min: x_max+1, y_min:y_max+1,z_min:z_max+1]

        data_cube_path = os.path.join(save_dir, clump_id_wcs + '.fits')
        if os.path.exists(data_cube_path):
            os.remove(data_cube_path)
        fits.writeto(data_cube_path, core_data)


def get_falseclumps(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir):
    """
    将每个云核切出来，放入立方体数据块中，保存为fits文件到save_dir文件夹
    :param mask_name: 云核检测掩模
    :param loc_outcat_name: 局部像素核表
    :param loc_outcat_wcs_name: 局部wcs核表
    :param origin_fit:原始云核数据
    :param save_dir: 切出来的小立方体保存的文件夹
    :return:
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    mask_data = fits.getdata(mask_name)  # 每个cell的mask
    outcat_loc = pd.read_csv(loc_outcat_name, sep='\t')
    outcat_loc_wcs = pd.read_csv(loc_outcat_wcs_name, sep='\t')
    origin_data = fits.getdata(origin_fit)
    false_index=list(range(outcat_loc.shape[0]))
    if len(false_index)>=1000:
        false_index=random.sample(false_index,1000)
    for i in false_index:
        clump_i = outcat_loc.iloc[i]
        clump_i_wcs = outcat_loc_wcs.iloc[i]
        clump_id_wcs = clump_i_wcs['ID']
        clump_id = clump_i['ID']

        core_x = np.array(np.where(mask_data == clump_id)[0])
        core_y = np.array(np.where(mask_data == clump_id)[1])
        core_z = np.array(np.where(mask_data == clump_id)[2])
        x_min = core_x.min()
        x_max = core_x.max()
        y_min = core_y.min()
        y_max = core_y.max()
        z_min = core_z.min()
        z_max = core_z.max()

        mask_copy = mask_data.copy()
        mask_copy[mask_copy != clump_id] = 0
        mask_copy = mask_copy / clump_id

        multiply_data = origin_data * mask_copy  # mask与原始数据相乘
        core_data = multiply_data[x_min: x_max+1, y_min:y_max+1,z_min:z_max+1]

        data_cube_path = os.path.join(save_dir, clump_id_wcs + '.fits')
        if os.path.exists(data_cube_path):
            os.remove(data_cube_path)
        fits.writeto(data_cube_path, core_data)

if __name__ == '__main__':
    #R2
    #云核
    p = Pool(10)
    # r2 5rms
    # detect_dir = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/First_detect/R2_200_detect'
    # cell_list = [i for i in os.listdir(detect_dir)]
    # mwisp_path = r'/home/data/clumps_share/MWISP/0_R2_Real_Data/R2_200_L'
    # save_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/R2_200_L'
    # os.makedirs(save_path, exist_ok=True)
    # for i, item in enumerate(cell_list):
    #     mask_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_mask.fits')
    #     loc_outcat_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat.csv')
    #     loc_outcat_wcs_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat_wcs.csv')
    #     origin_fit = os.path.join(os.path.join(mwisp_path, item.rstrip('_L')), item + '.fits')
    #     save_dir = os.path.join(os.path.join(save_path, item))
    #     p.apply_async(get_clumps, args=(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir))
    # p.close()
    # p.join()


    # 扣负样本的核
    # detect_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/detect_clumpfind'
    # match_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/match_clumpfind'
    # cell_list = [i for i in os.listdir(match_dir)]
    # cell_list=[i for i in cell_list if i not in os.listdir('/home/data/longc/pycharm/voxnet/R2_R16_3/R2/False_clump')]
    # mwisp_path = r'/home/data/clumps_share/MWISP/0_R2_Real_Data/R2_200_L'
    # save_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/False_clump'
    # os.makedirs(save_path,exist_ok=True)
    # for i, item in enumerate(cell_list):
    #     mask_name = os.path.join(os.path.join(detect_dir, item), 'out.fits')
    #     loc_outcat_name = os.path.join(os.path.join(match_dir, item), 'False_table/new_False.txt')
    #     loc_outcat_wcs_name = os.path.join(os.path.join(match_dir, item), 'False_table/false_outcat_wcs.csv')
    #     origin_fit = os.path.join(os.path.join(mwisp_path, item.rstrip('_L')), item + '.fits')
    #     save_dir = os.path.join(os.path.join(save_path, item))
    #     get_falseclumps(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir)
        # p.apply_async(get_clumps, args=(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir))
    # p.close()
    # p.join()




    # p = Pool(20)
    # # r16 5rms
    # detect_dir = r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_detect'
    # cell_list = [i for i in os.listdir(detect_dir)]
    # mwisp_path = r'/home/data/clumps_share/MWISP/0_R16_Real_data/R16_200_L'
    # save_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_200_L'
    # os.makedirs(save_path,exist_ok=True)
    # for i, item in enumerate(cell_list):
    #     mask_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_mask.fits')
    #     loc_outcat_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat.csv')
    #     loc_outcat_wcs_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat_wcs.csv')
    #     origin_fit = os.path.join(os.path.join(mwisp_path, item.rstrip('_L')), item + '.fits')
    #     save_dir = os.path.join(os.path.join(save_path, item))
    #     p.apply_async(get_clumps, args=(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir))
    # p.close()
    # p.join()

    #r16 3rms
    detect_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_data1/data/R16/R16_LDC_detect'
    cell_list = [i for i in os.listdir(detect_dir)]
    mwisp_path = r'/home/data/clumps_share/MWISP/0_R16_Real_data/R16_200_L'
    save_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_3rms_clump'
    os.makedirs(save_path,exist_ok=True)
    for i, item in enumerate(cell_list):
        mask_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_mask.fits')
        loc_outcat_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat.csv')
        loc_outcat_wcs_name = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat_wcs.csv')
        origin_fit = os.path.join(os.path.join(mwisp_path, item.rstrip('_L')), item + '.fits')
        save_dir = os.path.join(os.path.join(save_path, item))
        p.apply_async(get_clumps, args=(mask_name, loc_outcat_name, loc_outcat_wcs_name, origin_fit, save_dir))
    p.close()
    p.join()

