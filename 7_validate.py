import numpy as np
import tensorflow as tf
import textCNN_with_elmo as tc
import os
import csv
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers
from tensorflow.contrib import learn
from datetime import datetime
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from config import *
from func_lib import *


def validate():
    print("Validating with validation set...")
    # load validation set
    print("Loading validation set...")
    valid_labels, valid_text_ids = load_labels(
        valid_label_path, label_index, num_classes, one_hot=False)
    valid_texts = load_texts(valid_data_path, valid_text_ids)
    valid_size = len(valid_texts)
    tokenized_valid_texts = tokenize(valid_texts)

    # Create a TokenBatcher to map text to token ids
    batcher = TokenBatcher(train_vocab_file)
    
    # restore the TextCNN model
    print("Restoring TextCNN model...")
    tf.reset_default_graph()
    cnn_path = rootdir + r"models\TextCNN_with_elmo\4-classifier\\" \
               + r"label" + str(label_index) \
               + r"\TextCNN_with_elmo"
    cnn_path_dir = os.path.dirname(cnn_path)
    meta_file_name = os.listdir(cnn_path_dir)[-1]
    meta_path = os.path.join(cnn_path_dir, meta_file_name)

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(cnn_path)))

    graph = tf.get_default_graph()
    input_tensors = {'t_real_text': graph.get_tensor_by_name('real_text_input:0'),
                     'input_y': graph.get_tensor_by_name('input_y:0')}
    feedback = {'loss': graph.get_tensor_by_name('loss/add:0'),
                'accuracy': graph.get_tensor_by_name('accuracy/accuracy:0')}
    features = graph.get_tensor_by_name('g_conv/Squeeze:0')

    # Extracting features of texts in validation set
    print("Extracting features of texts in validation set(4 classes)...")
    all_features = []
    batch_num = valid_size // batch_size
    for batch_no in range(batch_num):
        try:
            batch_texts = tokenized_valid_texts[
                batch_no * batch_size: (batch_no + 1) * batch_size]
        except IndexError:
            batch_texts = tokenized_valid_texts[batch_no * batch_size: ]
        batch_texts = batcher.batch_sentences(batch_texts)
        batch_texts = pad_and_cut(batch_texts, MAX_LEN)
            
        batch_features = sess.run(
            features, feed_dict={input_tensors['t_real_text']: batch_texts})
        all_features.append(batch_features)
    valid_features = np.vstack(all_features)
    
    sess.close()
    
    # restore the 4-class-svm model
    print("Restoring 4-class svm and predicting labels...")
    svm_path = rootdir + r"models\SVM\4-classifier\\" \
                 + r"\svm_label" + str(label_index) + r".m"
    clf = joblib.load(svm_path)

    # predict labels of validation set with svm
    print("Predicting labels of validation set with svm...")
    predicted = clf.predict(valid_features)
    print(classification_report(valid_labels, predicted))
    
    acc = accuracy_score(valid_labels, predicted)
    f11 = f1_score(valid_labels, predicted, average=None)
    f12 = f1_score(valid_labels, predicted, average='macro')
    print("f11=", f11)
    print("f12=", f12)
    print("acc=", acc)


if __name__ == '__main__':
    global label_index
    global rootdir
    global cnn_path
    global svm_path
    label_index = 0
    validate()
