import os
import numpy as np
from MCVnet import ResNet
from deal_data_1.get_tfrecord import scale_tf
import astropy.io.fits as fits
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import colors


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpus = tf.config.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_pic0(x, file_path):
    sub_fig_num = x.shape[-1]
    fig = plt.figure(figsize=[8, 2.72])
    ii = 0
    pad = 0.01
    wth = 0.32
    for i in range(3):
        for j in range(1):
            ax = fig.add_axes([pad * (i + 1) + wth * i, pad * 8 / 2.72 * (j + 1) + j * wth, wth, wth*8/2.72])
            temp = x[0, :, :, :, 0].sum(ii)
            ax.imshow(temp, origin='lower')
            ii += 1
            ax.set_yticks([])
            ax.set_xticks([])
            ax.minorticks_off()
    fig.savefig(os.path.join(file_path, 'x.png'), dpi=200, format='png')
    # plt.show()
    plt.close(fig)


def get_pic(stem, file_path):
    sub_fig_num = stem.shape[-1]
    num_ = sub_fig_num // 4
    fig = plt.figure(figsize=[8, 8])
    ii = 0
    pad = 0.01
    wth = 0.2375
    for i in range(4):
        for j in range(num_):
            ax = fig.add_axes([pad * (i + 1) + wth * i, pad * (j + 1) + j * wth, wth, wth])
            temp = stem[0, :, :, :, ii].sum(0)
            ax.imshow(temp, origin='lower')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.minorticks_off()
            ii += 1
    fig.savefig(os.path.join(file_path, 'stem10.png'), dpi=200, format='png')
    plt.close(fig)
    # plt.show()


def get_pic11(stem_all, file_path, kernel=1):
    stem_all_1 = stem_all[..., kernel]
    sub_fig_num = stem_all_1[:16,...].shape[0]
    num_ = sub_fig_num // 4
    fig = plt.figure(figsize=[8.5, 8])
    ii = 0
    pad = 0.01
    wth = 0.2375
    wth1 = 0.2375 * 8 / 8.5
    ax_list = []
    vmax = stem_all_1.sum(axis=1).max()
    vmin = stem_all_1.sum(axis=1).min()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    idx_ = [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3]
    for i in range(4):
        for j in range(num_):
            ax = fig.add_axes([pad * (i + 1) + wth1 * i, pad * (j + 1) + j * wth, wth1, wth])
            ax_list.append(ax)
            temp = stem_all_1[idx_[ii], :, :, :].sum(0)

            hh = ax.imshow(temp, origin='lower', norm=norm)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.minorticks_off()
            ax.text(0.5, 0.5, 'Epoch=%d' % (idx_[ii] + 1), fontsize=13, color='red')
            ii += 1

    fig.subplots_adjust(right=8/8.2)

    # colorbar 左 下 宽 高
    l = 8/8.5
    b = 0.01
    w = 0.015
    h = 1 - 2 * b

    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(hh, cax=cbar_ax)
    # a = ax.pcolormesh(temp, norm=norm, cmap=plt.get_cmap('rainbow'))
    # fig.colorbar(a, ax=ax_list, shrink=0.5)
    fig.savefig(file_path, dpi=300, format='png')
    plt.close(fig)
    # plt.show()


def get_pic1(res_b, file_path):
    sub_fig_num = res_b.shape[-1]
    num_ = sub_fig_num // 8
    fig = plt.figure(figsize=[8, 4.02])
    ii = 0
    pad = 0.005
    wth = 0.119375
    for i in range(8):
        for j in range(num_):
            ax = fig.add_axes([pad * (i + 1) + wth * i, pad * 8 / 4.02 * (j + 1) + j * wth * 8 / 4.02, wth, wth * 8 / 4.02])
            temp = res_b[0, :, :, :, ii].sum(0)
            ax.imshow(temp, origin='lower')

            ax.set_yticks([])
            ax.set_xticks([])
            ax.minorticks_off()
            ii += 1
    fig.savefig(os.path.join(file_path, 'res_b0.png'), dpi=200, format='png')
    plt.close(fig)


def get_pic2(conv3d, file_path):
    sub_fig_num = conv3d.shape[-1]
    num_ = sub_fig_num // 8
    fig = plt.figure(figsize=[8, 8])
    ii = 0
    pad = 0.005
    wth = 0.119375
    for i in range(8):
        for j in range(num_):
            ax = fig.add_axes([pad * (i + 1) + wth * i, pad * 8 / 8 * (j + 1) + j * wth, wth, wth * 8 / 8])
            temp = conv3d[0, :, :, :, ii].sum(0)
            ii += 1
            ax.imshow(temp, origin='lower')

            ax.set_yticks([])
            ax.set_xticks([])
            ax.minorticks_off()
    fig.savefig(os.path.join(file_path, 'conv3d0.png'), dpi=200, format='png')
    plt.close(fig)

if __name__ == '__main__':
    model = ResNet([1])
    model.build(input_shape=(None, 30, 30, 30, 1))
    model_path = r'../models/model.h5'
    model.load_weights(model_path)
    # model_path = r'/home/data/longc/pycharm/voxnet/R2_R16_3/training_result/date_0516_bg'
    # model = load_model(model_path)
    data_path = r'../data/example_data'
    for item in os.listdir(data_path):
        if not item.endswith('.fits'):
            continue
        data_path_ = os.path.join(data_path, item)
        file_path = data_path_.replace('.fits', '')
        os.makedirs(file_path, exist_ok=True)
        data = fits.getdata(data_path_)
        scale = 30
        data_v = scale_tf(data, scale)
        x = tf.cast(data_v, tf.float32)
        # y = tf.keras.utils.to_categorical(y, num_classes=2)
        # y = tf.squeeze(y)
        x = tf.expand_dims(x, -1)
        x = tf.expand_dims(x, 0)
        y= model.predict(x)
        stem = model.stem.predict(x)
        res_b = model.layer1.predict(stem)
        conv3d = model.layer2.predict(res_b)

        get_pic(stem, file_path)
        get_pic0(x.numpy(), file_path)
        get_pic1(res_b, file_path)
        get_pic2(conv3d, file_path)

