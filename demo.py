# -*- coding: utf-8 -*-
# @Time : 2023/8/27 22:01
# @Author : longchen
# @File : demo.py
# @Project : MCVnet

from deal_data_1.detect_R16  import test

model_path=r'models/model'
data_path=r'data/example_data'
save_path=r'data'
test(model_path,data_path,save_path)

if __name__ == '__main__':
    pass