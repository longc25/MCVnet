# -*- coding: utf-8 -*-
# @Time : 2022/11/14 18:02
# @Author : longchen
# @File : clumpfind.py
# @Project : voxnet_M16_test

import numpy as np
import pandas as pd
from starlink import wrapper, kappa, convert, cupid

wrapper.change_starpath("/home/share/star-2018A")
import astropy.io.fits as fits
import os
from multiprocessing import Pool
import time
import shutil, glob
from multiprocessing import Pool


def SFellWalker(path_data, out_path, find_outcat, RMS):
    cupid.findclumps(path_data, out=out_path, outcat=find_outcat, method='fellwalker', deconv=False,
                     rms=RMS, backoff='False', WCSPAR='False', config='^fellwalker_para.txt')
    core_table = fits.getdata(find_outcat)

    return core_table


def SGaussclumps(path_data, out_path, find_outcat, RMS):
    cupid.findclumps(path_data, out=out_path, outcat=find_outcat, method='gaussclumps', deconv=False,
                     rms=RMS, backoff='False', WCSPAR='False', config='^gaussclumps_para.txt')
    core_table = fits.getdata(find_outcat)

    return core_table


def SReinhold(path_data, out_path, find_outcat, RMS):
    cupid.findclumps(path_data, out=out_path, outcat=find_outcat, method='reinhold', deconv=False,
                     rms=RMS, backoff='False', WCSPAR='False', config='^reinhold_para.txt')
    core_table = fits.getdata(find_outcat)

    return core_table


def SClumpFind(path_data, out_path, find_outcat, RMS):
    cupid.findclumps(in_=path_data, out=out_path, outcat=find_outcat, method='clumpfind', deconv=False,
                     rms=RMS, backoff='False', WCSPAR='False', config='^clumpfind_para.txt')
    core_table = fits.getdata(find_outcat)

    return core_table


def get_foutcat(path_data, out_path, find_outcat, RMS, mask_path, outcat_savepath):
    start = time.time()
    core_table = SClumpFind(path_data, out_path, find_outcat, RMS)
    convert.ndf2fits(out_path, mask_path)
    Cen1 = core_table['Cen1'] + 0.5
    # print(len(Cen1))
    Cen2 = core_table['Cen2'] + 0.5
    Cen3 = core_table['Cen3'] + 0.5
    #     center = np.c_[Cen1,Cen2,Cen3]
    Size1 = core_table['Size1']
    Size2 = core_table['Size2']
    Size3 = core_table['Size3']
    #     size=np.c_[x_size,y_size,z_size]
    Sum = core_table['Sum']
    Peak1 = core_table['Peak1'] + 0.5
    Peak2 = core_table['Peak2'] + 0.5
    Peak3 = core_table['Peak3'] + 0.5
    Peak = core_table['Peak']
    Volume = core_table['Volume']
    index_id = np.array(range(1, len(Peak1) + 1, 1))
    if len(Cen1) == 0:
        d_outcat = []
    else:
        d_outcat = np.hstack(
            [[index_id, Peak1, Peak2, Peak3, Cen1, Cen2, Cen3, Size1, Size2, Size3, Peak, Sum, Volume]]).T
    df_outcat = pd.DataFrame(d_outcat,
                             columns=['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2',
                                      'Size3', 'Peak', 'Sum', 'Volume'])
    df_outcat.to_csv(outcat_savepath, sep='\t', index=False)


# 计算fpr
def get_fpr(match_result):
    match_path = os.path.join(match_result, 'Match_table')
    false_path = os.path.join(match_result, 'False_table')
    miss_path = os.path.join(match_result, 'Miss_table')

    row_df_match = 0  # 所有的匹配核表中的PSNR
    for parents, dirnames, filenames in os.walk(match_path):
        for filename in filenames:
            df1 = pd.read_csv(os.path.join(parents, filename), sep='\t')
            row_df_match += df1.shape[0]  # 合成一个大的dataframe

    row_df_false = 0
    for parents, dirnames, filenames in os.walk(false_path):
        for filename in filenames:
            df2 = pd.read_csv(os.path.join(parents, filename), sep='\t')
            row_df_false += df2.shape[0]

    row_df_miss = 0  # 所有的漏检核表中的PSNR
    for parents, dirnames, filenames in os.walk(miss_path):
        for filename in filenames:
            df3 = pd.read_csv(os.path.join(parents, filename), sep='\t')
            row_df_miss += df3.shape[0]
    print('match:%.2f,false:%.2f,miss:%.2f' % (row_df_match, row_df_false, row_df_miss))
    precision = (row_df_match / (row_df_false + row_df_match))  # 准确率
    recall = (row_df_match / (row_df_miss + row_df_match))  # 召回率
    f1 = 2 * precision * recall / (precision + recall)  # F1
    print('precision:%.2f,recall:%.2f,f1:%.2f' % (precision, recall, f1))


# def get_MatchFrame(ddf1, ddf2, usecols=None, mindt=2.0, dbias=3600):
#     df1 = ddf1.copy()
#     df2 = ddf2.copy()
#     if not isinstance(df1, pd.DataFrame):
#         raise RuntimeError('>>> Input df1 must be pandas DataFrame!')
#     if not isinstance(df2, pd.DataFrame):
#         raise RuntimeError('>>> Input df2 must be pandas DataFrame!')
#     if usecols is None:
#         usecols = ['RAJ2000', 'DEJ2000']
#     else:
#         usecols = list(usecols)
#
#     for tmclm in usecols:
#         if tmclm not in df1.columns or tmclm not in df2.columns:
#             raise RuntimeError('>>> Column {} not in frame!'.format(tmclm))
#     # # sort
#     # df1.sort_values(by=usecols, inplace=True, ignore_index=True)
#     # df2.sort_values(by=usecols, inplace=True, ignore_index=True)
#
#     # create KDTree
#     mcoors = df2[usecols].values.copy()
#     index2 = np.arange(len(mcoors))
#     ocoors = df1[usecols].values.copy()
#     index1 = np.arange(len(ocoors))
#
#     myKDtree = cKDTree(mcoors)
#     dists, indexs = myKDtree.query(ocoors)
#     dists *= dbias
#     # create distance matrix
#     indexArr = np.vstack((dists, indexs))
#     indexArr = np.vstack((indexArr, index1)).T
#
#     # selection step 1, distance cut
#     dists = indexArr[:, 0]
#     _midx = np.argwhere(dists < mindt)
#     if len(_midx) > 0:
#         _midx = _midx[:, 0]
#     else:
#         return [None, None, None, None]
#     sdistarr = indexArr[_midx, :]
#     sdistarr = pd.DataFrame(data=sdistarr, columns=['dist', 'mids', 'oids'])
#     sdistarr1 = sdistarr.groupby('mids').filter(lambda x: len(x) <= 1)
#     sdistarr1.reset_index(drop=True, inplace=True)
#     sdistarr2 = sdistarr.groupby('mids').filter(lambda x: len(x) > 1)
#
#     # find duplicated matches
#     if len(sdistarr2) > 0:
#         sdistarr2.sort_values(by=['mids', 'dist'], inplace=True, ignore_index=True)
#         sdistarr2.reset_index(drop=True, inplace=True)
#         # _dupids = sdistarr2.groupby('mids').groups
#         sdistarr3 = sdistarr2.groupby('mids').head(1)
#         sdistarr1 = sdistarr1.append(sdistarr3, ignore_index=True)
#     else:
#         # no duplicated match
#         pass
#
#     _midx1 = sdistarr1['oids'].values.astype(np.int_)
#     _mdf1 = df1.iloc[_midx1].copy()
#     if len(_mdf1) == 0: _mdf1 = None
#     _noidx1 = np.setdiff1d(index1, _midx1)
#     _nomdf1 = df1.iloc[_noidx1].copy()
#     if len(_nomdf1) == 0: _nomdf1 = None
#
#     _midx2 = sdistarr1['mids'].values.astype(np.int_)
#     _mdf2 = df2.iloc[_midx2].copy()
#     if len(_mdf2) == 0: _mdf2 = None
#     _noidx2 = np.setdiff1d(index2, _midx2)
#     _nomdf2 = df2.iloc[_noidx2].copy()
#     if len(_nomdf2) == 0: _nomdf2 = None
#     return [_mdf1, _nomdf1, _mdf2, _nomdf2]
#
# # The Evaluation function
# def Evaluate(origin_center, com, origin_flux, detect_flux):
#     dist = 0
#     match_id_i = []
#     match_id_j = []
#     delta_dist = []
#     delta_flux = []
#     match_flux = []
#     dist_record = []
#     #     dist_record = [[],[]]
#     A = np.stack(origin_center, axis=0)
#     ddf1 = pd.DataFrame({'Cen1': A[:, 0], 'Cen2': A[:, 1], 'Cen3': A[:, 2]})
#     B = np.stack(com, axis=0)
#     ddf2 = pd.DataFrame({'Cen1': B[:, 0], 'Cen2': B[:, 1], 'Cen3': B[:, 2]})
#     _mdf1, _nomdf1, _mdf2, _nomdf2 = get_MatchFrame(ddf1, ddf2, usecols=['Cen1', 'Cen2', 'Cen3'], mindt=2.0,
#                                                     dbias=1)
#     match_id_i = list(_mdf1.index)
#     match_id_j = list(_mdf2.index)
#     for i, j in zip(match_id_i, match_id_j):
#         dist = ((origin_center[i][0] - com[j][0]) ** 2 + (origin_center[i][1] - com[j][1]) ** 2 + (
#                     origin_center[i][2] - com[j][2]) ** 2) ** (1 / 2)
#         #         dist_record[0].append(((origin_center[i][1]-com[j][1])**2 + (origin_center[i][2]-com[j][2])**2)**(1/2))
#         #         dist_record[1].append(origin_center[i][0]-com[j][0])
#         dist_record.append(dist)
#         delta_flux.append((detect_flux[j] - origin_flux[i]) / origin_flux[i])
#         match_flux.append(origin_flux[i])
#     TP = len(match_id_i)
#     FP = len(com) - TP
#     FN = len(origin_center) - TP
#     recall = TP / (TP + FN)
#     precision = TP / (TP + FP)
#     F1 = 2 * TP / (2 * TP + FP + FN)
#     return recall, precision, F1, dist_record, delta_flux, match_flux
#
# recall, precision, F1, dist_record, delta_flux, match_flux = Evaluate(origin_center, did_ConBased['clump_com'],
#                                                                       origin_flux, did_ConBased['clump_sum'])
# print('ConBased recall:', recall)
# print('ConBased precision:', precision)
# print('ConBased F1:', F1)
# print('ConBased Dist:', np.mean(dist_record))
# print('ConBased Flux:', np.mean(delta_flux))


if __name__ == '__main__':
    # 测试检测算法从哪个像素坐标开始的
    # save_dir = r'/home/data/longc/pycharm/voxnet/deal_data/Simulated_Data_0855-045_L'
    # os.makedirs(save_dir, exist_ok=True)
    # out_path = os.path.join(save_dir, 'out.sdf')
    # find_outcat = os.path.join(save_dir, 'outcat.FIT')
    # mask_path = os.path.join(save_dir, 'out.fits')
    # outcat_savepath = os.path.join(save_dir, 'outcat.csv')
    # path_data = r'/home/data/longc/pycharm/voxnet/deal_data/Simulated_Data_0855-045_L_1.fits'
    # rms = 0.5
    # core_table = SGaussclumps(path_data, out_path, find_outcat, rms)
    # print(core_table)
    # get_foutcat(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath)

    # 仿真数据
    # for t in range(0, 40):
    #     rms = 2
    #     save_dir = r'/home/data/longc/pycharm/voxnet/simulate_data/simulate_100/detect_dir3/%04d' % t
    #     os.makedirs(save_dir, exist_ok=True)
    #     path_data = r'/home/data/longc/pycharm/voxnet/simulate_data/simulate_100/3D_clump_100/out/gaussian_out_%04d.fits' % t
    #     out_path = os.path.join(save_dir, 'out.sdf')
    #     find_outcat = os.path.join(save_dir, 'outcat.FIT')
    #     mask_path = os.path.join(save_dir, 'out.fits')
    #     outcat_savepath = os.path.join(save_dir, 'outcat.csv')
    #     get_foutcat(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath)
    #
    # # p = Pool(20)
    # for t in range(0, 15):
    #     rms = 2
    #     save_dir = r'/home/data/longc/pycharm/voxnet/simulate_data/test_100/match_clumpfind3/detect/%04d' % t
    #     os.makedirs(save_dir, exist_ok=True)
    #     path_data = r'/home/data/longc/pycharm/voxnet/simulate_data/test_100/3D_clump_100/out/gaussian_out_%04d.fits' % t
    #     out_path = os.path.join(save_dir, 'out.sdf')
    #     find_outcat = os.path.join(save_dir, 'outcat.FIT')
    #     mask_path = os.path.join(save_dir, 'out.fits')
    #     outcat_savepath = os.path.join(save_dir, 'outcat.csv')
    #     get_foutcat(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath)
    #     # p.apply_async(get_foutcat, args=(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath))
    # p.close()
    # p.join()

    # 计算clumpfind的准确率、召回率和f1
    # match_result = r'/home/data/longc/pycharm/voxnet/simulate_data/simulate_100/match_clumpfind3'  #40个团块
    # get_fpr(match_result)
    #
    # match_result1 = r'/home/data/longc/pycharm/voxnet/simulate_data/test_100/match_clumpfind3'  #15个团块
    # get_fpr(match_result1)

    # 处理R2和R16数据
    # R16
    # detect_dir = r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_detect'
    # cell_list = [i for i in os.listdir(detect_dir)]
    # mwisp_path = r'/home/data/clumps_share/MWISP/0_R16_Real_data/R16_200_L'
    # save_path = r'/home/data/longc/pycharm/voxnet/R2_16_data/R16/detect_clumpfind'
    # for i, item in enumerate(cell_list):
    #     outcat_path=os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat.csv')
    #     log_path=os.path.join(os.path.join(detect_dir, item),'LDC_auto_detect_log.txt')
    #     rms=float(open(log_path).readlines()[2].lstrip('the rms of data: '))
    #     df_outcat=pd.read_csv(outcat_path,sep='\t')
    #     if df_outcat.shape[0]==0:#跳过LDC检测不到的文件夹
    #         continue
    #     save_dir=os.path.join(os.path.join(save_path, item))
    #     os.makedirs(save_dir, exist_ok=True)
    #     path_data = os.path.join(os.path.join(mwisp_path, item.rstrip('_L')), item + '.fits')
    #     data = fits.getdata(path_data)
    #     new_path_data=os.path.join('/home/data/longc/pycharm/voxnet/R2_16_data/R16/mwisp_data', item + '.fits')
    #     fits.writeto(new_path_data,data)
    #
    #     out_path = os.path.join(save_dir, 'out.sdf')
    #     find_outcat = os.path.join(save_dir, 'outcat.FIT')
    #     mask_path = os.path.join(save_dir, 'out.fits')
    #     outcat_savepath = os.path.join(save_dir, 'loc_outcat.csv')
    #     get_foutcat(new_path_data, out_path, find_outcat, rms, mask_path, outcat_savepath)

    # header=fits.getheader('/home/data/clumps_share/MWISP/0_R16_Real_data/R16_200_L/1810-020/1810-020_L.fits')
    # print(header['RMS'])
    # #R2
    # p = Pool(30)
    # detect_dir = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/First_detect/R2_200_detect'
    # cell_list = [i for i in os.listdir(detect_dir)]
    # mwisp_path = r'/home/data/clumps_share/MWISP/0_R2_Real_Data/R2_200_L'
    # save_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/detect_clumpfind'
    # for i, item in enumerate(cell_list):
    #     outcat_path = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_loc_outcat.csv')
    #     log_path = os.path.join(os.path.join(detect_dir, item), 'LDC_auto_detect_log.txt')
    #     rms = float(open(log_path).readlines()[2].lstrip('the rms of data: '))
    #     df_outcat = pd.read_csv(outcat_path, sep='\t')
    #     if df_outcat.shape[0] == 0:  # 跳过LDC检测不到的文件夹
    #         continue
    #
    #     save_dir = os.path.join(os.path.join(save_path, item))
    #     # if os.path.exists(save_dir):
    #     #     shutil.rmtree(save_dir)
    #     # else:
    #     os.mkdir(save_dir)
    #     path_data = os.path.join(os.path.join(mwisp_path, item.rstrip('_L')), item + '.fits')
    #
    #     out_path = os.path.join(save_dir, 'out.sdf')
    #     find_outcat = os.path.join(save_dir, 'outcat.FIT')
    #     mask_path = os.path.join(save_dir, 'out.fits')
    #     outcat_savepath = os.path.join(save_dir, 'loc_outcat.csv')
    #     get_foutcat(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath)
        # p.apply_async(get_foutcat, args=(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath))
    # p.close()
    # p.join()

    # 第二象限数据
    # p = Pool(20)
    # mwisp_path = r'/home/data/clumps_share/MWISP/0_G100+00/200_L'
    # cell_list = [i for i in os.listdir(mwisp_path)]
    # cell_list.sort()
    # save_path = r'/home/data/clumps_share/data_longc/0_G100+00/detect_clumpfind_2'
    # for i, item in enumerate(cell_list[0:21]):
    #     save_dir=os.path.join(os.path.join(save_path, item+'_L'))
    #
    #     os.makedirs(save_dir, exist_ok=True)
    #     path_data = os.path.join(os.path.join(mwisp_path, item), item + '_L.fits')
    #     head =fits.getheader(path_data)
    #     keys=list(head.keys())
    #     # print(keys)
    #     if 'RMS' in keys:
    #         rms = head[' RMS']
    #         # print(rms)
    #     else:
    #         rms_path=os.path.join(os.path.join(mwisp_path, item), item + '_L_rms.fits')
    #         data = fits.getdata(rms_path)
    #         rms =  np.median(data)
    #     print(rms)
    #     out_path = os.path.join(save_dir, 'out.sdf')
    #     find_outcat = os.path.join(save_dir, 'outcat.FIT')
    #     mask_path = os.path.join(save_dir, 'out.fits')
    #     outcat_savepath = os.path.join(save_dir, 'loc_outcat.csv')
    #     get_foutcat(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath)
        # p.apply_async(get_foutcat, args=(path_data, out_path, find_outcat, rms, mask_path, outcat_savepath))
    # p.close()
    # p.join()


    save_path = r'/home/data/clumps_share/data_longc/G190-195/match_clumpfind'
    cell_list = [i for i in os.listdir(save_path)]
    count=0
    for i, item in enumerate(cell_list):
        save_dir = os.path.join(os.path.join(save_path, item ))
        outcat_savepath = os.path.join(save_dir, 'False_table/new_False.txt')
        if not os.path.exists(outcat_savepath):
            continue
        df_outcat=pd.read_csv(outcat_savepath,sep='\t')
        count+=df_outcat.shape[0]
    print(count)