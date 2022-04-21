from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences=word2vec.Text8Corpus('../pretreatment/x_train')
model=word2vec.Word2Vec(sentences, sg=1, window=5, min_count=5, negative=3, sample=0.001, hs=1)
model.save('../model/wv.model')

