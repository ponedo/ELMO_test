import os

# text & label loading
label_index = 0 # from 0 to 19
num_classes = 4

MAX_LEN = 200
emb_dim = 200
batch_size = 100

relativity_weight = 5

# data paths
rootdir = r"G:\\\\project2"
train_data_path = os.path.join(rootdir, 'data', 'trainingset_data.txt')
train_label_path = os.path.join(rootdir, 'data', 'trainingset_label.txt')

valid_data_path = os.path.join(rootdir, 'data', 'validationset_data.txt')
valid_label_path = os.path.join(rootdir, 'data', 'validationset_label.txt')

testa_data_path = os.path.join(rootdir, 'data', 'testa_data.txt')

testb_data_path = os.path.join(rootdir, 'data', 'testb_data.txt')

stopwords_path = os.path.join(rootdir, 'data', 'stopwords.txt')

merged_data_path = os.path.join(rootdir, 'data', 'data.txt')
merged_label_path = os.path.join(rootdir, 'data', 'label.txt')

# building bilm
wv_path = os.path.join(rootdir, 'build', 'premodel_+minc2_n3_sg0_hs1.txt')
train_bilm_vocab_file = os.path.join(rootdir, 'build', 'premodel_+minc2_n3_sg0_hs1.txt')

# bilm
bilm_options_file = os.path.join(rootdir, 'build/checkpoints', 'options.json')
bilm_weight_file = os.path.join(rootdir, 'build', 'elmo_weights.hdf5')
train_vocab_file = os.path.join(rootdir, 'build', 'train_vocab.txt')
whole_vocab_file = os.path.join(rootdir, 'build', 'whole_vocab.txt')
token_embedding_file = os.path.join(rootdir, 'build', 'vocab_embeddings.hdf5')

# model path
cnn_path = os.path.join(rootdir, 'models', 'TextCNN_with_elmo',
                        str(num_classes) + '-classifier',
                        'label' + str(label_index),
                        'TextCNN_with_elmo')

svm_path = os.path.join(rootdir, 'models', 'SVM',
                        str(num_classes) + '-classifier',
                        'svm_label' + str(label_index) + '.m')

# result path
result_path = os.path.join(rootdir, 'results', 'result_label' + str(label_index))

final_result_path = os.path.join(rootdir, 'results', 'final_result.csv')
