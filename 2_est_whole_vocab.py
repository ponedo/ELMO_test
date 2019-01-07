import collections
from config import *
from operator import itemgetter


def est_vocab(corpus_file_path, vocab_file_path):
    print("Establishing vocabulary with corpus \"%s\"" % corpus_file_path)
    print("Please make sure that the corpus is preprocessed(segmented, stopwords removed and labels extracted)")
    counter = collections.Counter()
    total_num = 0
    fi = open(corpus_file_path, "r", encoding='utf-8')

    flag = 1
    for line in fi.readlines():
        # skip the first line (column head)
        if flag:
            flag = 0
            continue

        # read the line
        words = line.strip().split()

        # count the words in the line except stopwords and write the segmented line
        for word in words:
            counter[word] += 1
            total_num += 1

    fi.close()

    print("Establishing vocabulary...")
    sorted_counter = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_counter]

    sorted_words = ["<S>", "</S>", "<UNK>"] + sorted_words
##    if len(sorted_words) > VOCAB_SIZE:
##        sorted_words = sorted_words[:VOCAB_SIZE]

    with open(vocab_file_path, "w", encoding='utf-8') as v:
        for word in sorted_words:
            if word not in ["<S>", "</S>", "<UNK>"]:
                v.write('%s\n' % word)
            else:
                v.write(word + '\n')


def est_univocab(pos_vocab_path, neg_vocab_path, univocab_path):
    f_pos = open(pos_vocab_path, 'r', encoding='utf-8')
    f_neg = open(neg_vocab_path, 'r', encoding='utf-8')

    pos_info = [line.strip().split() for line in f_pos.readlines()]
    vocab_pos = {item[0]: float(item[2]) for item in pos_info}

    neg_info = [line.strip().split() for line in f_neg.readlines()]
    vocab_neg = {item[0]: float(item[2]) for item in neg_info}

    f_pos.close()
    f_neg.close()

    unique_vocab_pos = {}
    unique_vocab_neg = {}

    for word, value in vocab_pos.items():
        unique_vocab_pos[word] = value - vocab_neg.get(word, 0)
    for word, value in vocab_neg.items():
        unique_vocab_neg[word] = value - vocab_pos.get(word, 0)

    sorted_unique_vocab_pos = sorted(unique_vocab_pos.items(), key=lambda x: x[1], reverse=True)
    sorted_unique_vocab_neg = sorted(unique_vocab_neg.items(), key=lambda x: x[1], reverse=True)

    fu_pos = open(univocab_path + "_1.vocab", 'w', encoding='utf-8')
    fu_neg = open(univocab_path + "_-1.vocab", 'w', encoding='utf-8')

    print("Establishing positive vocabulary...")
    for word, value in sorted_unique_vocab_pos:
        if value > 0:
            fu_pos.write("%s\t%f\n" % (word, value))
        else:
            break
    print("Establishing negative vocabulary...")
    for word, value in sorted_unique_vocab_neg:
        if value > 0:
            fu_neg.write("%s\t%f\n" % (word, value))
        else:
            break
    fu_pos.close()
    fu_neg.close()


def word_to_id(div_data_path, vocab_path, sentences_output_path):
    # substitute words in segmented dataset with their id in the vocabulary
    print("substitute words in segmented dataset with their id in the vocabulary...")
    with open(vocab_path, "r", encoding='utf-8') as v:
        vocab = {word.strip(): value for value, word in enumerate([line.split()[0] for line in v.readlines()])}
        v.close()

    fi = open(div_data_path, 'r', encoding='utf-8')
    fo = open(sentences_output_path, 'w', encoding='utf-8')

    for line in fi.readlines():
        words_in_line = line.strip().split() + ["<eos>"]
        fo.write(" ".join(
            [str(vocab[word] if word in vocab else vocab["<unk>"]) for word in words_in_line])+'\n')

    fi.close()
    fo.close()
    

if __name__ == '__main__':
    est_vocab(merged_data_path, whole_vocab_file)
