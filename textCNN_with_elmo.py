import tensorflow as tf
import numpy as np
import math
from utils import *
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers


def conv2d(input_,
           output_dim,
           k_w,
           MAX_LEN,
           k_h=2, d_h=1, d_w=1, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, 1, output_dim],   #[2,200,1,200]
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        pooled_T = tf.nn.max_pool(conv,ksize=[1,MAX_LEN - 2 - 2 +1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
        T_flat = tf.squeeze(pooled_T)
        return T_flat


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev,seed=0))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


class TextCNN_with_elmo:
    '''
    OPTIONS
    text_length : Max Text Len 200
    emb_dim : word embed dimension 200
    batch_size : Batch Size 50
    num_classes : num of classes 2
    bilm_options_file : bilm_options_file
    bilm_weight_file : bilm_weight_file
    token_embedding_file : token_embedding_file
    l2_bilm_lambda : l2_bilm_lambda None
    l2_cnn_lambda : l2_cnn_lambda 0.0

    '''

    def __init__(self, options):
        self.options = options

    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def textCNN(self, t_text):
        emb_dim = self.options['emb_dim']
        t_text_expanded = tf.expand_dims(t_text, -1)
        t_text_conv = conv2d(
            t_text_expanded,
            self.options['emb_dim'],
            self.options['emb_dim'],
            MAX_LEN = self.options['text_length'],
            name = 'g_conv')

        return t_text_conv

    def build_model(self):
        t_real_text = tf.placeholder('int32', shape=(None, self.options['text_length']), name='real_text_input')
        input_y = tf.placeholder('int32', [None, self.options['num_classes']], name="input_y")

        bilm = BidirectionalLanguageModel(
            self.options['bilm_options_file'],
            self.options['bilm_weight_file'],
            use_character_inputs=False,
            embedding_weight_file=self.options['token_embedding_file'])

        bilm_layers_result = bilm(t_real_text)

        elmo_text = weight_layers(
            'elmo_text',
            bilm_layers_result,
            l2_coef=self.options['l2_bilm_lambda'])

        print("elmo_text.shape and emb_dim:")
        print(elmo_text['weighted_op'].shape)
        print(self.options['emb_dim'])
                
        t_text_conv = self.textCNN(elmo_text['weighted_op'])
        l2_loss = tf.constant(0.0)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.options['emb_dim'], self.options['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.options['num_classes']]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(t_text_conv, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            loss = tf.reduce_mean(losses) + \
                   self.options['l2_cnn_lambda'] * l2_loss \
                   + tf.add_n(elmo_text['regularization_op'])

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        input_tensors = {
            't_real_text' : t_real_text,
            'input_y' : input_y
        }

        feedback = {
            'loss' : loss,
            'accuracy' : accuracy
        }

        features = t_text_conv
        
        return input_tensors, feedback, features


if __name__ == '__main__':
    model_options = {
            'text_length': 200,
            'emb_dim': 200,
            'batch_size': 100,
            'emb_matrix': [[0 for _ in range(100)]],
            'vocab_size': 1,
            'num_classes': 4,
            'l2_reg_lambda': 0.0
            }
    cnn = TextCNN_with_elmo(model_options)
    _, _, _ = cnn.build_model()

    # for v in [n.name for n in tf.get_default_graph().as_graph_def().node]:
    #    print(v)

