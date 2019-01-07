import numpy as np
import tensorflow as tf
import textCNN_with_elmo as tc
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers
from tensorflow.contrib import learn
from datetime import datetime
from config import *
from func_lib import *


# train
epoch_num = 10
learning_rate = 0.00005
beta1 = 0.5
l2_bilm_lambda = 0,0
l2_cnn_lambda = 0.0


def train_cnn(first_use=True):

    print("Loading texts and labels...")
    # load training set
    train_labels, train_text_ids = \
                  load_labels(train_label_path, label_index, num_classes, one_hot=True)
    train_texts = load_texts(train_data_path, train_text_ids)
    train_size = len(train_texts)
    tokenized_train_texts = tokenize(train_texts)

    # Create a TokenBatcher to map text to token ids
    batcher = TokenBatcher(train_vocab_file)

    if first_use:

        # build TextCNN model
        tf.reset_default_graph()
        model_options = {
            'text_length': MAX_LEN,
            'emb_dim': emb_dim,
            'batch_size': batch_size,
            'num_classes': num_classes,
            'bilm_options_file': bilm_options_file,
            'bilm_weight_file': bilm_weight_file,
            'token_embedding_file': token_embedding_file,
            'l2_bilm_lambda': l2_bilm_lambda,
            'l2_cnn_lambda': l2_cnn_lambda
            }

        print("Building TextCNN model")
        cnn = tc.TextCNN_with_elmo(model_options)
        input_tensors, feedback, features = cnn.build_model()
        
        optim = tf.train.AdamOptimizer(learning_rate, beta1)\
                .minimize(feedback['loss'], name='optim')

        saver = tf.train.Saver(max_to_keep=1)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        epoch_done = 0

    else:
        tf.reset_default_graph()
        cnn_path_dir = os.path.dirname(cnn_path)
        meta_file_name = os.listdir(cnn_path_dir)[-1]
        epoch_done = int(meta_file_name[18:-5])
        meta_path = os.path.join(cnn_path_dir, meta_file_name)

        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(cnn_path)))

        graph = tf.get_default_graph()

        input_tensors = {'t_real_text': graph.get_tensor_by_name('real_text_input:0'),
                     'input_y': graph.get_tensor_by_name('input_y:0')}
        feedback = {'loss': graph.get_tensor_by_name('loss/add:0'),
                    'accuracy': graph.get_tensor_by_name('accuracy/accuracy:0')}
        optim = graph.get_operation_by_name('optim')

        # Reset max_to_keep
        saver = tf.train.Saver(max_to_keep=1)

    # train
    for epoch in [(i + epoch_done + 1) for i in range(epoch_num)]:
        print("<Epoch no. %d>" % epoch)
        # train text cnn model
        print("Training text cnn model...")
        start = datetime.now()
        sum_accuracy = 0.0
        total_loss = 0.0
        
        batch_num = train_size // batch_size
        for batch_no in range(batch_num):
            if batch_no % (batch_num // 10) == 0:
                print("%d0%%" % (batch_no // (batch_num // 10)))
                
            batch_texts = tokenized_train_texts[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_texts = batcher.batch_sentences(batch_texts)
            batch_texts = pad_and_cut(batch_texts, MAX_LEN)

            batch_labels = train_labels[batch_no * batch_size : (batch_no + 1) * batch_size]

            _, loss, accuracy = \
               sess.run([optim, feedback['loss'], feedback['accuracy']],
                        feed_dict={input_tensors['t_real_text']: batch_texts,
                                   input_tensors['input_y']: batch_labels})

            sum_accuracy += accuracy
            total_loss += loss
            
        avg_accuracy = sum_accuracy / batch_num

        # save the cnn model
        if not os.path.exists(os.path.dirname(cnn_path)):
            os.mkdir(os.path.dirname(cnn_path))
        saver.save(sess, cnn_path, global_step=epoch)

        end = datetime.now()
        print("TextCNN training complete!")
        print("Train report: loss: %d, accuracy: %d, train time: %f" %
              (total_loss, avg_accuracy, (end - start).seconds))
    sess.close()


if __name__ == '__main__':
    global label_index
    global num_classes
    global rootdir
    global cnn_path
    num_classes = 4
    for i in range(0, 20):
        train_cnn(first_use=True)
        label_index += 1
        cnn_path = rootdir + r"models\TextCNN_with_elmo\\" \
                   + str(num_classes) + r"-classifier\\" \
                   + r"label" + str(label_index) \
                   + r"\TextCNN_with_elmo"
        break
