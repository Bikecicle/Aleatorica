from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, LeakyReLU, Flatten, AveragePooling2D,\
    Conv2D, Reshape, add, UpSampling2D
from math import log2

import tensorflow.keras.backend as K
import numpy as np
import random

from layers import *

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))


def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def upsample(x):
    return K.resize_images(x, 2, 2, "channels_last", interpolation='bilinear')


def block_res(block_idx):
    return 2 ** (block_idx + 2)


def layer_name(block_idx, layer_type, idx):
    return 'block' + str(block_idx) + '_' + layer_type + str(idx)


class StyleGAN(object):

    def __init__(self,
                 latent_size=128,
                 channels=2,
                 n_blocks=7,
                 init_res=4,
                 kernel_size=3,
                 kernel_initializer='he_uniform',
                 padding='same',
                 n_fmap=None):

        # Variables
        self.latent_size = latent_size
        self.channels = channels
        self.n_blocks = n_blocks
        self.init_res = init_res
        self.final_res = init_res * (2 ** (n_blocks - 1))
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.n_fmap = n_fmap

    def gen_block(self, x, latent, noise, block):

        # First modulation
        style = Dense(units=x.shape[-1],
                      kernel_initializer=self.kernel_initializer,
                      name='block{}_style0')(latent)
        delta = Dense(units=self.n_fmap[block],
                      kernel_initializer='zeros',
                      name='block{}_delta0')(noise)

        x = Conv2DMod(filters=self.n_fmap[block],
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      kernel_initializer=self.kernel_initializer,
                      name='block{}_conv0'.format(block))([x, style])
        x = add([x, delta])
        x = LeakyReLU()(x)

        # Second modulation
        style = Dense(units=x.shape[-1],
                      kernel_initializer=self.kernel_initializer,
                      name='block{}_style1')(latent)
        delta = Dense(units=self.n_fmap[block],
                      kernel_initializer='zeros',
                      name='block{}_delta1')(noise)

        x = Conv2DMod(filters=self.n_fmap[block],
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      kernel_initializer=self.kernel_initializer,
                      name='block{}_conv1'.format(block))([x, style])
        x = add([x, delta])
        x = LeakyReLU()(x)

        # To rgb
        style = Dense(units=x.shape[-1],
                      kernel_initializer=self.kernel_initializer,
                      name='block{}_style2')(latent)

        r = Conv2DMod(filters=self.channels,
                      kernel_size=1,
                      kernel_initializer=self.kernel_initializer,
                      name='block{}_to_rgb'.format(block))([x, style])

        scale = 2 ** (self.n_blocks - block - 1)
        r = UpSampling2D(size=[scale, scale])(r)

        return x, r

    def build_gen(self):

        # Inputs
        latent_in = []
        noise_in = []
        for i in range(self.n_blocks * 2):
            latent_in.append(Input([self.latent_size], name='style_input' + str(i)))
            res = block_res(i // 2)
            noise_in.append(Input(shape=[res, res, 1], name='noise_input' + str(i)))

        # Latent
        seed_in = Input(shape=[1], name='latent_seed')
        x = Dense(self.init_res * self.init_res * self.n_fmap[0],
                  activation='relu',
                  kernel_initializer='random_normal',
                  name='latent_const')(seed_in)
        x = Reshape(target_shape=[self.init_res, self.init_res, self.n_fmap[0]],
                    name='latent_reshape')(x)

        rgb = []

        x, r = self.gen_block(x, latent_in[0], noise_in[0], 0)
        rgb.append(r)

        for i in range(1, self.n_blocks):
            x = UpSampling2D()(x)
            x, r = self.gen_block(x, latent_in[i], noise_in[i], i)
            rgb.append(r)

        x = add(rgb)

        gen = Model(inputs=latent_in + noise_in + [seed_in], outputs=x, name='gen')
        return gen

    def dis_block(self, x, block):

        res = Conv2D(filters=self.n_fmap[block],
                     kernel_size=1,
                     kernel_initializer=self.kernel_initializer)(x)

        x = Conv2D(filters=self.n_fmap[block],
                   kernel_size=self.kernel_size,
                   padding=self.padding,
                   kernel_initializer=self.kernel_initializer)(res)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.n_fmap[block],
                   kernel_size=self.kernel_size,
                   padding=self.padding,
                   kernel_initializer=self.kernel_initializer)(x)
        x = LeakyReLU()(x)

        x = add([res, x])

        return x

    def build_dis(self):

        img_in = Input(shape=[self.final_res, self.final_res, self.channels], name='img_input')
        x = img_in

        for i in range(self.n_blocks - 1):
            x = self.dis_block(x, i)
            x = AveragePooling2D()(x)

        x = self.dis_block(x, self.n_blocks - 1)
        x = Flatten()(x)

        x = Dense(1, kernel_initializer='he_uniform')(x)

        return Model(inputs=img_in, outputs=x, name='dis')

    def random_latent(self, n):
        return np.random.normal(0.0, 1.0, size=[n, self.latent_size]).astype('float32')

    def random_latent_list(self, n):
        n_layers = self.n_blocks * 2
        return [self.random_latent(n)] * n_layers

    def mixed_latent_list(self, n):
        n_layers = self.n_blocks * 2
        tt = int(random.random() * n_layers)
        p1 = [self.random_latent(n)] * tt
        p2 = [self.random_latent(n)] * (n_layers - tt)

        return p1 + [] + p2

    def random_noise(self, n):
        noise = []
        for i in range(self.n_blocks * 2):
            res = block_res(i // 2)
            noise.append(np.random.uniform(0.0, 1.0, size=[n, res, res, 1]).astype('float32'))
        return noise
