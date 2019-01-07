from gensim.models import Word2Vec
from config import merged_data_path

sg = 0
size = 100
window = 5
min_count = 2
negative = 3
sample = 0.001
hs = 1


if __name__ == '__main__':
    f = open(merged_data_path, 'r', encoding='utf-8')
    docs=[]

    for sentence in f.readlines():
        cis = sentence.strip().split()
        docs.append(cis)
    
    model = Word2Vec(
        docs,
        sg=sg,
        size=size,
        window=window,
        min_count=min_count,
        negative=negative,
        sample=sample,
        hs=hs)
    
    model.wv.save_word2vec_format(
        r'build/premodel_+' +
        r'minc' + str(min_count) +
        r'_n' + str(negative) +
        r'_sg' + str(sg) +
        r'_hs' + str(hs) +
        r'.txt',
        binary=False)

    f.close()
