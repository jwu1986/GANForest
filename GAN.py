# -*- coding:utf-8 -*-
# J.W
# May 2021 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.cluster import KMeans
import numpy as np
import time
import datetime
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default = 'select_featur6_500.csv', type = str, help = 'input file')
opt = parser.parse_args()


def make_generator_model():
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


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, use_bias=False, input_shape=[outputLength, ], dtype='float32'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, use_bias=False, dtype='float32'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, use_bias=False, dtype='float32'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, use_bias=False, dtype='float32'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, use_bias=False, dtype='float32'))
    return model


def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]

    t = tf.random.normal([batchsz, outputLength])
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)

    grads = tape.gradient(d_interplote_logits, interplate)

    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp


# 判别损失 判别器要做两件事情，既要真的趋近于1，又要假的趋近于0
def discriminator_loss(real_output, fake_output, batch_x):
    d_loss_real = celoss_ones(real_output)
    d_loss_fake = celoss_zeros(fake_output)
    gp = gradient_penalty(discriminator, batch_x, fake_output)

    loss = d_loss_fake + d_loss_real + 1. * gp
    return loss, gp


# 生成损失  生成器使得假的趋近于1
def generator_loss(fake_output):
    loss = celoss_ones(fake_output)
    # print(fake_output,loss)
    return loss


# 单步训练
# 注意 `tf.function` 的使用
# 该注解使函数被“编译”
@tf.function
def train_step(ECG):
    global generator_optimizer, discriminator_optimizer, total_optimizer
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    noise = tf.cos(4 * noise)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as total_tape:
        generated_ECG = generator(noise, training=True)
        real_output = discriminator(ECG, training=True)
        fake_output = discriminator(generated_ECG, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss, gp = discriminator_loss(real_output, fake_output, ECG)
        total_loss = tf.tanh(tf.abs(gen_loss) - tf.abs(disc_loss))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_total = total_tape.gradient(total_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    total_optimizer.apply_gradients(zip(gradients_of_total, generator.trainable_variables))
    return gen_loss, disc_loss, gp, total_loss


# 定义训练
def train(dataset, epochs, loss_pd, ca):
    global train_index
    for epoch in range(epochs):
        start = time.time()
        cunt = 0
        for image_batch in dataset:
            cunt += 1
            gen_loss, disc_loss, gp, total_loss = train_step(image_batch)
            loss_pd.loc[train_index] = [ca, gen_loss.numpy(), disc_loss.numpy(), gp.numpy(), total_loss.numpy()]
            train_index += 1
            print('\rgen {} dis {} GP {} tol {}'.format(gen_loss, disc_loss, gp, total_loss), end='')
        print('\nTime for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


if __name__ == '__main__':

    older_data = pd.read_csv(opt.input, header=None,low_memory=False)
    print('load data...')

    inputLength = 1000
    older_data = older_data.iloc[1:, :]
    older_data.to_csv("temp.csv", index=False)
    older_data = pd.read_csv("temp.csv", header=None)
    #print(older_data)

    x = older_data.iloc[:, 2:]

    outputLength = len(x.columns)
    BATCH_SIZE = 20
    # 定义训练
    EPOCHS = 100
    noise_dim = inputLength
    # 训练损失记录
    loss_pd = pd.DataFrame(columns=['label', 'gen_loss', 'disc_loss', 'gp', 'total_loss'])
    train_index = 0

    cate_data = list(pd.Categorical(older_data[1]).categories)
    print('-------------------------------------------')
    print(cate_data)
    print('-------------------------------------------')
    # 定义优化器
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,amsgrad=False)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,amsgrad=False)
    total_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,amsgrad=False)

    for ca in cate_data:
        # 获取训练数据
        train_dataset = older_data[older_data.iloc[:, 1] == ca]
        train_dataset = train_dataset.iloc[:, 2:]
        train_dataset.reset_index(drop=True, inplace=True)
        train_size = int(np.ceil(len(train_dataset) / BATCH_SIZE) * BATCH_SIZE)
        train_lack = train_size - len(train_dataset)
        length_data = len(train_dataset)
        for i in range(train_lack):
            train_dataset.loc[length_data + i] = train_dataset.loc[np.random.choice(train_dataset.index)]
        train_data = tf.cast(train_dataset, 'float32')
        train_datasets = tf.data.Dataset.from_tensor_slices(train_data).shuffle(train_size).batch(BATCH_SIZE)
        # 初始化模型
        generator = make_generator_model()
        discriminator = make_discriminator_model()
        # 进行训练
        train(train_datasets, EPOCHS, loss_pd, ca)
        # 保存
        generator.save_weights('G_model_{}.hdf5'.format(ca))
        discriminator.save_weights('D_model_{}.hdf5'.format(ca))
    filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + 'train_loss.txt'
    loss_pd.to_csv(filename, index=False)