import numpy as np
import tensorflow as tf
import textCNN_with_elmo as tc
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers
from tensorflow.contrib import learn
from datetime import datetime
from sklearn import svm
from sklearn.externals import joblib
from config import *
from func_lib import *


def train_svm():
    
    print("Loading texts and labels...")
    # load training set
    train_labels, train_text_ids = \
                  load_labels(train_label_path, label_index, num_classes, one_hot=False)
    train_texts = load_texts(train_data_path, train_text_ids)
    train_size = len(train_texts)
    tokenized_train_texts = tokenize(train_texts)

    # Create a TokenBatcher to map text to token ids
    batcher = TokenBatcher(train_vocab_file)
    
    # restore the TextCNN model
    print("Restoring TextCNN model...")
    tf.reset_default_graph()
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

    # train svm
    # extract features of texts in training set first
    print("Extracting features of texts in training set with trained text cnn model...")
    all_features = []
    batch_num = train_size // batch_size
    for batch_no in range(batch_num):
        try:
            batch_texts = tokenized_train_texts[
                batch_no * batch_size: (batch_no + 1) * batch_size]
        except IndexError:
            batch_texts = tokenized_train_texts[batch_no * batch_size: ]
        batch_texts = batcher.batch_sentences(batch_texts)
        batch_texts = pad_and_cut(batch_texts, MAX_LEN)
        
        batch_features = sess.run(
            features, feed_dict={input_tensors['t_real_text']: batch_texts})
        all_features.append(batch_features)
    train_features = np.vstack(all_features)
    
    sess.close()
    
    # train svm with training set features and labels
    print("Training svm with training set features and labels...")
    start = datetime.now()
    clf = svm.SVC(kernel='linear', C=1, class_weight='balanced')
    clf.fit(train_features, train_labels)

    # save svm model
    if not os.path.exists(os.path.dirname(svm_path)):
        os.mkdir(os.path.dirname(svm_path))

    joblib.dump(clf, svm_path)
    
    end = datetime.now()
    print("SVM training complete!")
    print("Train report: train time: %f" % (end-start).seconds)


if __name__ == '__main__':
    global label_index
    global num_classes
    global rootdir
    global cnn_path
    global svm_path
    num_classes = 4
    for i in range(0, 20):
        train_svm()
        label_index += 1
        cnn_path = rootdir + r"models\TextCNN_with_elmo\\" \
                   + str(num_classes) + r"-classifier\\" \
                   + r"label" + str(label_index) \
                   + r"\TextCNN_with_elmo"
        svm_path = rootdir + r"models\SVM\\" \
           + str(num_classes) +  r"-classifier\\" \
           + r"\svm_label" + str(label_index) + r".m"
