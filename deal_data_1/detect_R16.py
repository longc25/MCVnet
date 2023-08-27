# -*- coding: utf-8 -*-
# @Time : 2023/4/13 17:42
# @Author : longchen
# @File : detect_R16.py
# @Project : voxnet_M16_test


import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.io.fits as fits
from multiprocessing import Pool
import pandas as pd
from scipy import ndimage


np.random.seed(1)
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def per_fits_data(path):
    """
    做数据预处理,根据文件夹中文件的命名来确定标签
    验证通过，2019/11/17
    """
    scale=30
    filelist_ = os.listdir(path)
    X = []
    for i, item in enumerate(filelist_):
        item_path=os.path.join(path, item)
        temp = fits.getdata(item_path)#读取fits文件

        # print(temp.shape)
        # 做局部归一化
        # temp = (temp - temp.min()) / (temp.max() - temp.min())
        # 计算缩放因子
        scaling_factor = np.array([scale / i for i in temp.shape]).min()
        # 体积归一化
        if max(temp.shape) >=scale:
        # 体积归一化
            temp = ndimage.zoom(temp, (scaling_factor, scaling_factor, scaling_factor))
        [temp_x, temp_y, temp_z] = temp.shape
        pad_x, pad_y, pad_z = (scale - temp_x) // 2, (scale - temp_y) // 2, (scale - temp_z) // 2
        temp1 = np.pad(temp, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant')
        [temp_x, temp_y, temp_z] = temp1.shape
        temp1 = np.pad(temp1, ((scale - temp_x, 0), (scale - temp_y, 0), (scale - temp_z, 0)), 'constant')

        bg_data = fits.getdata(r'data/bg.fits')
        temp1 = temp1 + bg_data
        temp1 = (temp1 - temp1.min()) / (temp1.max() - temp1.min())

        X.append(temp1)

    X = np.array(X)
        # print(X.shape)
    return X


def test(model_path,data_path,save_path):
    '''

    :param model_path: 模型的地址，文件格式
    :param data_path: 预测数据的文件夹
    :param save_path: 预测结果保存地址，在csv里面
    :return:
    '''
    # model = load_model(r'/home/data/longc/pycharm/voxnet/realdata_result/R2_R16/date_0309/model/')
    model = load_model(model_path)
    # path=r'/home/data/longc/pycharm/voxnet/R2_16_data/R16/uncertain_sample/'
    X_predicate= per_fits_data(data_path)
    # X_predicate = tf.convert_to_tensor(X_predicate)
    X_predicate = tf.expand_dims(X_predicate, -1)
    X_predicate = tf.cast(X_predicate, dtype=tf.float32)
    print(X_predicate.shape)
    filelist_ = os.listdir(data_path)
    filelist_=[i.rstrip('.fits') for i in filelist_]
    ypreds=model.predict(X_predicate)
    # print(ypreds)
    # Y_test_class = np.argmax(ypreds, axis=1)
    # # print(Y_test_class)
    confidence=ypreds[:,1]
    dict_result={'ID':filelist_,'Confidence':confidence}
    df_test_pred=pd.DataFrame(dict_result)#所有的结果
    df_test_pred.to_csv(os.path.join(save_path,'all_data_result.csv'), sep='\t', index=False)

    # df_realclump=df_test_pred[df_test_pred['Test_result']==1]   #是核
    # df_realclump.to_csv(os.path.join(save_path,'yes.csv'),sep='\t',index=False)
    #
    # df_nonclump=df_test_pred[df_test_pred['Test_result'] == 0]  #不是核
    # df_nonclump.to_csv(os.path.join(save_path, 'no.csv'), sep='\t', index=False)


#对比3rms和5rms r16
def compare_clump(ldc_3rms_detect,ldc_5rms_detect,match_save_dir):
    # ldc_3rms= r'/home/data/longc/pycharm/voxnet/R2_R16_data1/data/R16/R16_LDC_detect'
    cell_list = [i for i in os.listdir(ldc_5rms_detect)]
    # ldc_5rms= r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_detect'
    # match_save_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_data1/data/R16/match_ldc_35'
    os.makedirs(match_save_dir,exist_ok=True)
    for i, item in enumerate(cell_list):
        # print(item)
        ldc_outcat_3rms = os.path.join(os.path.join(ldc_3rms_detect, item), 'LDC_auto_loc_outcat.csv')
        df_ldc_3rms = pd.read_csv(ldc_outcat_3rms, sep='\t')

        ldc_outcat_5rms = os.path.join(os.path.join(ldc_5rms_detect, item), 'LDC_auto_loc_outcat.csv')
        df_ldc_5rms = pd.read_csv(ldc_outcat_5rms, sep='\t')
        if df_ldc_5rms.shape[0] != 0 and df_ldc_3rms.shape[0] == 0:#若3和5倍的rms都为空，则不匹配
            # print(item)
            # continue
        # match_save_path = os.path.join(match_save_dir, item)
        # os.makedirs(match_save_path, exist_ok=True)
        # match_detect(ldc_outcat_5rms, ldc_outcat_3rms, match_save_path)

#统计mcvnet预测结果all_data_result.csv中为按照概率划分的
def get_yes(pre_dir,no_threshold,yes_threshold,mode='Yes'):
    # 计算resnet_csv的长度
    df_pre_all = pd.DataFrame()
    for i, cell in enumerate(os.listdir(pre_dir)):
        savepath_cell = os.path.join(pre_dir, cell)
        pre_csv=os.path.join(savepath_cell,'all_data_result.csv')
        df_pre=pd.read_csv(pre_csv,sep='\t')
        df_pre_all=pd.concat([df_pre_all,df_pre])
    df_pre_all.drop_duplicates(subset=['ID'],keep='first',inplace=True)
    df_pre_all.to_csv('/home/data/clumps_share/data_longc/R16_LDC_data/verify/3d_resnet_result.csv',index=False,sep='\t')
    if mode=='No':
        df_yes_all = df_pre_all[df_pre_all['Confidence'] < no_threshold]  # 小于某个阈值为no
    elif mode=='Uncertain':
        df_yes_all = df_pre_all[(df_pre_all['Confidence'] >= no_threshold)&(df_pre_all['Confidence'] <=yes_threshold)]  # 小于某个阈值为no
    else:
        df_yes_all = df_pre_all[df_pre_all['Confidence'] > yes_threshold]     #大于某个阈值为yes

    count_yes=df_yes_all.shape[0]
    print(count_yes)

    return df_yes_all


#人工证认与LDC_5rms匹配
def verify_resnet(verify_path,pre_dir,no_threshold,yes_threshold,save_path,mode='Yes'):
    df_verify=pd.read_csv(verify_path,sep='\t')
    df_yes_all=get_yes(pre_dir,no_threshold,yes_threshold,mode)
    df_merge=pd.merge(df_verify,df_yes_all,on='ID',how='inner')
    difference_list=[i for i in df_verify['ID'].values if i not in df_yes_all['ID'].values]
    ps_df=pd.Series(difference_list)
    # ps_df.to_csv('/verify/df.csv',index=False,sep='\t')
    df_merge.to_csv(save_path,index=False,sep='\t')
    print(difference_list,df_merge.shape[0])

def cal_accuracy(TP,FP,FN,TN):
    return (TP+TN)/(TP+TN+FP+FN)


#画统计分布直方图
# 画置信度曲线
def hist_confidence(pre_dir):
    df_pre_all = pd.DataFrame()
    for i, cell in enumerate(os.listdir(pre_dir)):
        savepath_cell = os.path.join(pre_dir, cell)
        pre_csv = os.path.join(savepath_cell, 'all_data_result.csv')
        df_pre = pd.read_csv(pre_csv, sep='\t')
        df_pre_all = pd.concat([df_pre_all, df_pre])
    df_pre_all.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    confidence = df_pre_all['Confidence']
    fig = plt.figure(figsize=(15, 8))
    font_size = 14
    ax= fig.add_axes([0.1, 0.1, 0.88, 0.88])
    plt.xlabel('Confidence', fontsize=font_size)
    plt.ylabel('number', fontsize=font_size)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=14, top='on', right='on')
    ax.hist(confidence, bins=40,histtype='bar', alpha=0.5)
    # plt.legend(loc='best')
    plt.grid(linestyle=':')
    plt.show()


if __name__ == '__main__':
    # r16
    # 测试 5_rms
    # fits_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_5rms_clump'
    # cell_list = os.listdir(fits_dir)
    # model_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/training_result/date_0417/model/'
    # save_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_5rms_predict_0417/'
    # os.makedirs(save_path,exist_ok=True)
    # # 计算resnet_csv的长度
    # df_csv_all = pd.DataFrame()
    # for i, cell in enumerate(cell_list):
    #     cell_path = os.path.join(fits_dir, cell)
    #     if len(os.listdir(cell_path)) == 0:
    #         continue
    #     savepath_cell = os.path.join(save_path, cell)
    #     os.makedirs(savepath_cell, exist_ok=True)
    #     test(model_path, cell_path, savepath_cell)
    # hist_confidence(save_path,0.5)


    #测试 3_rms
    # fits_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_3rms_clump'
    # cell_list=os.listdir(fits_dir)
    # model_path=r'/home/data/longc/pycharm/voxnet/R2_R16_3/training_result/date_0417/model/'
    # save_path=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_3rms_predict_0417/'
    # os.makedirs(save_path, exist_ok=True)
    # #计算resnet_csv的长度
    # df_csv_all=pd.DataFrame()
    # for i,cell in enumerate(cell_list):
    #     cell_path = os.path.join(fits_dir, cell)
    #     if len(os.listdir(cell_path)) == 0:
    #         continue
    #     savepath_cell=os.path.join(save_path,cell)
    #     os.makedirs(savepath_cell,exist_ok=True)
    #     test(model_path,cell_path,savepath_cell)
    # hist_confidence(save_path)

    #匹配3rms与5rms
    # ldc_3rms_detect=r'/home/data/longc/pycharm/voxnet/R2_R16_data1/data/R16/R16_LDC_detect'
    # ldc_5rms_detect=r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_detect'
    # match_save_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/match_LDC_35'
    # compare_clump(ldc_3rms_detect, ldc_5rms_detect, match_save_dir)

    #人工认证与resnet对比 5rms
    # verify_path=r'/home/data/clumps_share/data_longc/云核认证/R16_LDC/second_verify/Yes_LDC_R16_3.csv'
    # verify_path=r'/home/data/clumps_share/data_longc/云核认证/R16_LDC/second_verify/No_LDC_R16_3.csv'
    # verify_path = r'/home/data/clumps_share/data_longc/云核认证/R16_LDC/second_verify/Uncertain_LDC_R16_3.csv'
    # pre_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_5rms_predict_0417'
    # confidence_threshold=0.5
    # verify_resnet(verify_path,pre_dir,confidence_threshold,'No')
    #计算准确率
    # TP=1430
    # FP=355
    # FN=4
    # TN=13
    # print(cal_accuracy(TP, FP,FN,TN))

    # 人工认证与resnet对比 3rms
    verify_path=r'/home/data/clumps_share/data_longc/R16_LDC_data/verify/Yes_LDC_3.csv'
    # verify_path=r'/home/data/clumps_share/data_longc/R16_LDC_data/verify/No_LDC_3.csv'
    # verify_path = r'/home/data/clumps_share/data_longc/R16_LDC_data/verify/Uncertain_LDC_3.csv'
    # verify_path=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/3_5_between.csv'
    pre_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_3rms_predict_0417'
    no_threshold=0.3
    yes_threshold=0.5
    save_path=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/3_9.csv'
    verify_resnet(verify_path,pre_dir,no_threshold,yes_threshold,save_path,'Yes')
    # 计算准确率
    # TP=
    # FP=
    # FN=0
    # TN=0
    # print(cal_accuracy(TP, FP,FN,TN))

    #3rm检测出的云核中3~5rms之间的数量


    #mcvnet认为3rms中云核的统计情况
    # pre_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/R16_3rms_predict'
    # detect_3rms=r'/home/data/longc/pycharm/voxnet/R2_R16_data1/data/R16/R16_LDC_detect'
    # df_yes_all=get_yes(pre_dir, 0.5)
    # yes_id=df_yes_all['ID']
    # match_save_dir=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R16/match_LDC_35'
    # cell_list=os.listdir(match_save_dir)
    # for i,item in enumerate(cell_list[0]):
    #     match_table=os.path.join(os.path.join(match_save_dir,item),'Match_table/Match.txt')
    #     df_match=pd.read_csv(match_table,sep='\t')
    #     match_id=df_match['f_ID']
    #     outcat_5rms=os.path.join(os.path.join(detect_3rms,item),'LDC_auto_loc_outcat.csv')
    #     df_outcat=pd.read_csv(outcat_5rms,sep='\t')
    #     outcat_index=df_outcat[df_outcat['ID']  in  match_id].index.tolist()
    #     wcs_5rms = os.path.join(os.path.join(detect_3rms, item), 'LDC_auto_loc_outcat_wcs.csv')
