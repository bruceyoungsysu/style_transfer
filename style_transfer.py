from load_vgg import VGG
import numpy as np
import tensorflow as tf
import os

def resize_image(image):
    pass

class StyleTransfer:
    def __init__(self, content_image, style_image, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.content_image = resize_image(content_image)
        self.style_image = resize_image(style_image)

        # create CNN variables
        self.content_layer = 'conv4_2'
        self.style_layer = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # weights in calculating loss
        self.style_layers_w = [0.5, 1.0, 1.5, 3.0, 4.0]
        self.conten_w = 0.01
        self.style_w = 1

        self.lr = 2.0

    def load_input(self):
        self.input_img = tf.get_variable('input_img', shape=(1, self.image_width, self.image_height, 3),
                                         dtype=tf.float32, initializer=tf.zeros_initializer())

    def load_vgg(self):
        # load vgg network and initialize content img and style img
        self.vgg = VGG(self.input_img)
        self.vgg.build()
        self.content_image -= self.vgg.mean_pixels
        self.style_iamge -= self.vgg.mean_pixels

    def content_loss(self, P, F):
        """
        :param F:  content representation of content layer
        :param P:  content representation of generated layer
        :return:  content loss function
        """
        self.content_loss = (1/P.size) * tf.reduce_sum(F**2 + P**2)

    def gram_matrix(self, F, N, M):
        """
        :param F: feature map
        :param N:  the third dimension
        :param M: the product of the first two dimensions
        :return:
        """
        new_F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(new_F), new_F)

    def single_style_loss(self, a, g):
        """
        :param a: gram matrix of style image
        :param g: gram matrix of generated image
        :return:
        """
        N = g.shape[3]
        M = g.shape[0] * g.shape[1]
        G = self.gram_matrix(a, N, M)
        A = self.gram_matrix(g, N, M)
        return (1/(4*N**2*M**2))*tf.reduce_sum(tf.matmul(G-A,G-A))

    def style_loss(self, As):
        """
        :param As: feature representation of style layer
        :return: style loss function
        """
        loss = [self.single_style_loss(As[i], getattr(self.vgg, self.style_layer[i]) ) for i in range(len(As))]
        self.style_loss = sum([self.style_w[i]]*loss[i] for i in range(len(As)))

    def total_loss(self):
        with tf.name_scope('loss') as scope:
            with tf.Session() as sess:
                setattr(self.input_img, self.content_image)
                conten_gen_img = getattr(self.vgg, self.content_layer)
                conten_conten_img = sess.run(conten_gen_img)
                self.content_loss(conten_conten_img, conten_gen_img)

            with tf.Session() as sess:
                setattr(self.input_img, self.style_image)
                layers_style = [sess.run(getattr(self.vgg, style_layer)) for style_layer in self.style_layer]
                self.style_loss(layers_style)

            self.total_loss = self.conten_w*self.content_loss + self.style_w*self.style_loss

    def optimize(self):
        with tf.name_scope('optimize') as scope:
            pass

    def build(self):
        pass

    def train(self):
        pass