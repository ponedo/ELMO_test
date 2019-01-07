import gensim
from config import *

if __name__=="__main__":
    model=gensim.models.KeyedVectors.\
           load_word2vec_format(fname=wv_path, binary=False)

    words=model.vocab
    with open(train_vocab_file, 'w', encoding='utf-8') as f:
        f.write('<S>')
        f.write('\n')
        f.write('</S>')
        f.write('\n')
        f.write('<UNK>')
        f.write('\n')    # bilm-tf 要求vocab有这三个符号，并且在最前面
        for word in words:
            f.write(word)
            f.write('\n')
