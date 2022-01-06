# -*- coding:utf-8 -*-
# J.W
# May 2021 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default = 'select_feature.csv', type = str, help = 'input file')
opt = parser.parse_args()


def make_generator_model(inputLength, outputLength):
    model = tf.keras.Sequential()
    model.add(layers.Dense(inputLength * 2, use_bias=False, input_shape=(inputLength,), dtype='float32'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(inputLength * 4, use_bias=False, dtype='float32'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(inputLength * 4, use_bias=False, dtype='float32'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(outputLength, use_bias=False, activation='tanh', dtype='float32'))
    return model


def main():
    older_data = pd.read_csv(opt.input, header=None,low_memory=False)
    print(older_data)
    headerone = older_data.iloc[0:1,2:]
    #older_data = older_data.set_index("lable")
    
    older_data = older_data.iloc[1:, :]
    older_data.to_csv("temp.csv", index=False)
    older_data = pd.read_csv("temp.csv", header=None,low_memory=False)
    
    x = older_data.iloc[:, 2:]
    
    inputLength = 1000
    outputLength = len(x.columns)
    noise_dim = inputLength

    # 目标数据个数
    target_data_number = 3000
    print('target data number:', target_data_number)
    # 获取标签
    countclass = older_data.groupby(1).count()
    classes_dict = countclass[2].to_dict()
    print(classes_dict)

    for c in classes_dict:
        classes_dict[c] = target_data_number - classes_dict[c]
    new_data = pd.DataFrame()

    for ca in classes_dict:
        # 加载原数据
        train_dataset = older_data[older_data.iloc[:, 1] == ca]
        train_dataset = train_dataset.iloc[:, 2:]
        train_dataset.columns = range(1, outputLength + 1)
        train_dataset_de = train_dataset.describe()  # describe
        gan_new_data = pd.DataFrame([[ca]] * classes_dict[ca])
        # 初始化模型
        generator = make_generator_model(inputLength, outputLength)
        # 加载权重
        generator.load_weights('G_model_{}.hdf5'.format(ca))
        noise = tf.random.normal([classes_dict[ca], noise_dim])
        noise = tf.cos(4 * noise)
        print('generating {} : {}...'.format(ca, classes_dict[ca]))
        # 生成数据
        generated_ECG = generator(noise, training=False)
        generated_data = pd.DataFrame(generated_ECG.numpy(), columns=range(1, generated_ECG.shape[1] + 1))
        generated_data = generated_data * (train_dataset_de.loc['max'] - train_dataset_de.loc['min']) + \
                         train_dataset_de.loc['min']
        gan_new_data = pd.concat([gan_new_data, generated_data], axis=1)
        new_data = pd.concat([new_data, gan_new_data])
    print('generated GAN data...')
    older_data_copy = older_data.copy()
    older_data_copy = older_data_copy.iloc[1:, 1:]
    older_data_copy.columns = range(outputLength + 1)
    headerone = np.array(headerone).tolist()
    headerone[0].insert(0,'Label')
    new_data = pd.concat([older_data_copy, new_data])
    new_data = pd.DataFrame(np.array(new_data).tolist(),columns=headerone)
    new_data.to_csv('data_label_new.csv', index=False, header=True)


if __name__ == '__main__':
    main()
