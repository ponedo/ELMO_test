import numpy as np
from config import *


def load_labels(labels_path, label_index, num_classes, one_hot=False):
    f_label = open(labels_path, 'r')
    labels = []

    if num_classes == 3:
        text_ids = []
    else:
        text_ids = None

    if one_hot:
        for i, label in enumerate(f_label.readlines()):
            label = (label.strip().split(","))[label_index]
            if num_classes == 2:
                if label == '-2':
                    labels.append([1, 0])
                elif label in ['-1', '0', '1']:
                    labels.append([0, 1])
                else:
                    raise Exception("<Wrong label> label no." + str(i))
            elif num_classes == 3:
                if label == '-1':
                    labels.append([1, 0, 0])
                    text_ids.append(i)
                elif label == '0':
                    labels.append([0, 1, 0])
                    text_ids.append(i)
                elif label == '1':
                    labels.append([0, 0, 1])
                    text_ids.append(i)
                elif label == '-2':
                    continue
                else:
                    raise Exception("<Wrong label> label no." + str(i))
            elif num_classes == 4:
                if label == '-2':
                    labels.append([1, 0, 0, 0])
                elif label == '-1':
                    labels.append([0, 1, 0, 0])
                elif label == '0':
                    labels.append([0, 0, 1, 0])
                elif label == '1':
                    labels.append([0, 0, 0, 1])
                else:
                    raise Exception("<Wrong label> label no." + str(i))

    else:
        for i, label in enumerate(f_label.readlines()):
            label = (label.strip().split(","))[label_index]
            if num_classes == 2:
                if label == '-2':
                    labels.append(0)
                elif label in ['-1', '0', '1']:
                    labels.append(1)
                else:
                    raise Exception("<Wrong label> label no." + str(i))
            elif num_classes == 3:
                if label in ['-1', '0', '1']:
                    labels.append(int(label))
                    text_ids.append(i)
                elif label == '-2':
                    continue
                else:
                    raise Exception("<Wrong label> label no." + str(i))
            elif num_classes == 4:
                if label in ['-2', '-1', '0', '1']:
                    labels.append(int(label))
                else:
                    raise Exception("<Wrong label> label no." + str(i))
    f_label.close()
    return labels, text_ids


def load_texts(texts_path, text_ids=None):
    f_text = open(texts_path, 'r', encoding='utf-8')
    texts = f_text.readlines()
    f_text.close()
    if text_ids:
        texts = [texts[i] for i in text_ids]
    return texts


def tokenize(texts):
    print("Tokenizing")
    return [[token for token in text.strip().split()] for text in texts]


def pad_and_cut(text_ids, text_length):
    if text_ids.shape[1] >= text_length:
        return text_ids[:, :text_length]
    else:
        return np.pad(
            text_ids,
            ((0, 0), (0, text_length - text_ids.shape[1])),
            'constant',
            constant_values=(0, 0))
