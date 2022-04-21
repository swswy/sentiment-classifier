import numpy as np
import os.path
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, AveragePooling1D, SimpleRNN, Bidirectional
from keras.layers import Conv1D, MaxPool1D, Embedding, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import logging
from gensim.models import Word2Vec
from keras import optimizers
# 构造词向量字典
# 文本序号化Tokenizer
# 使用Embedding层将每个词编码转换为词向量
# 读取数据


logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

EMBEDDING_DIM = 100  # 词向量维度
max_document_length = 120


def load_file(filename, words_index):
    with open(os.path.join('..', 'pretreatment', 'x_' + filename), 'r', encoding='utf-8') as f:
        text = f.read()
    texts = text.split('\n')
    y = []
    with open(os.path.join('..', 'pretreatment', 'y_' + filename), 'r', encoding='utf-8') as f:
        for line in f:
            temp = [0, 0, 0, 0, 0, 0, 0]
            temp[int(line.strip())] = 1
            y.append(temp)
    return tokenizer(texts, words_index), np.array(y)


def get_word2vec_dictionaries():
    model = Word2Vec.load(os.path.join('..', 'model', 'cbow.model'))
    vocab_list = list(model.wv.key_to_index.keys())  # 存储 所有的 词语
    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典
    word_vector = {}  # 初始化`[word : vector]`字典
    embeddings_matrix = np.zeros((len(vocab_list) + 1, model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        # word_vector[word] = model.wv[word]  词语：词向量
        embeddings_matrix[i + 1] = model.wv[word]  # 词向量矩阵

    return word_index, embeddings_matrix


# texts是一个单词的列表的列表，表示很多句子
def tokenizer(texts, word2index):
    data = []
    for sentence in texts:
        new_txt = []
        new_sentence = [_ for _ in sentence.split(' ') if _]
        for word in new_sentence:
            try:
                new_txt.append(word2index[word])  # 把句子中的 词语转化为index
            except Exception:
                new_txt.append(0)

        data.append(new_txt)

    texts = pad_sequences(data, maxlen=max_document_length,
                          padding='post')  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return texts


def create_cnn(embeddings):
    # 输入就该是【0，1，2，5，】这种形式
    sequence_input = Input(shape=(max_document_length,))
    # EMBEDDING_DIM需要和之前预训练中词向量的dim一致
    embedding_layer = Embedding(input_dim=len(embeddings), output_dim=EMBEDDING_DIM, weights=[embeddings],
                                input_length=max_document_length)
    embedding_sequences = embedding_layer(sequence_input)
    # 100 120(max_document_length) 100（EMBEDDING_DIM)
    # 卷积核大小为3
    cnn1 = Conv1D(filters=30, kernel_size=3, activation='relu')(embedding_sequences)  # 100 118 40
    cnn1 = MaxPool1D(pool_size=3)(cnn1)  # 100 39 40
    cnn1 = Conv1D(filters=30, kernel_size=3, activation='relu')(cnn1)  # 100 37 40
    cnn1 = MaxPool1D(pool_size=3)(cnn1)  # 100 12 40
    cnn1 = Conv1D(filters=30, kernel_size=3, activation='relu')(cnn1)  # 100 10 40
    cnn1 = MaxPool1D(pool_size=3)(cnn1)  # 100 3 30
    cnn1 = Flatten()(cnn1)  # 100 90
    # 卷积核大小为4
    cnn2 = Conv1D(filters=30, kernel_size=4, activation='relu')(embedding_sequences)  # N 117 40
    cnn2 = MaxPool1D(pool_size=3)(cnn2)  # N 39 40
    cnn2 = Conv1D(filters=30, kernel_size=4, activation='relu')(cnn2)  # N 36 40
    cnn2 = MaxPool1D(pool_size=3)(cnn2)  # 100 12 40
    cnn2 = Conv1D(filters=30, kernel_size=4, activation='relu')(cnn2)  # N 9 40
    cnn2 = MaxPool1D(pool_size=3)(cnn2)  # N 3 40
    cnn2 = Flatten()(cnn2)  # N 90
    # 卷积核大小为5
    cnn3 = Conv1D(filters=30, kernel_size=5, activation='relu')(embedding_sequences)  # N 116 40
    cnn3 = MaxPool1D(pool_size=3)(cnn3)  # N 38 40
    cnn3 = Conv1D(filters=30, kernel_size=5, activation='relu')(cnn3)  # N 34 40
    cnn3 = MaxPool1D(pool_size=3)(cnn3)  # N 11 40
    cnn3 = Conv1D(filters=30, kernel_size=5, activation='relu')(cnn3)  # N 7 40
    cnn3 = MaxPool1D(pool_size=3)(cnn3)  # N 2 40
    cnn3 = Flatten()(cnn3)  # N 60

    # 合并
    merge = concatenate([cnn1, cnn2, cnn3], axis=1)
    # 全连接层
    x = Dense(60, activation='relu')(merge)
    # Dropout层
    x = Dropout(0.5)(x)
    # 输出层
    preds = Dense(7, activation='softmax')(x)
    # 定义模型
    cnn_model = Model(sequence_input, preds)
    cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0015), metrics=['acc'])
    return cnn_model


def create_mlp(embeddings):
    sequence_input = Input(shape=(max_document_length,))
    embedding_layer = Embedding(input_dim=len(embeddings), output_dim=EMBEDDING_DIM,
                                weights=[embeddings], input_length=max_document_length)
    embedding_sequences = embedding_layer(sequence_input)  # 128 120 100
    embedding_sequences = AveragePooling1D(pool_size=2)(embedding_sequences)
    flat = Flatten()(embedding_sequences)
    mlp = Dense(256, activation='relu')(flat)
    # mlp = Dense(64, activation='relu')(mlp)
    mlp1 = Dense(512, activation='relu')(flat)
    # mlp1 = Dense(128,activation='relu')(mlp1)
    merge = concatenate([mlp, mlp1], axis=1)
    merge = Dense(64, activation='relu')(merge)
    dropped = Dropout(0.35)(merge)
    output = Dense(7, activation='softmax')(dropped)
    mlp_model = Model(sequence_input, output)
    mlp_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0012), metrics=['acc'])
    return mlp_model


def create_rnn(embeddings):
    sequence_input = Input(shape=(max_document_length,))
    embedding_layer = Embedding(input_dim=len(embeddings), output_dim=EMBEDDING_DIM,
                                weights=[embeddings], input_length=max_document_length)
    embedding_sequences = embedding_layer(sequence_input)
    embedding_sequences = AveragePooling1D(pool_size=4)(embedding_sequences)
    lstm = LSTM(256, return_sequences=True)(embedding_sequences)
    lstm = LSTM(128, return_sequences=True)(lstm)
    lstm = LSTM(64)(lstm)
    # simple_rnn = SimpleRNN(256,dropout=0.35,recurrent_dropout=0.35)(embedding_sequences)
    # bi=Bidirectional(SimpleRNN(256,dropout=0.35,recurrent_dropout=0.5))(embedding_sequences)
    # merge = concatenate([lstm, simple_rnn], axis=1)

    densed = Dense(64, activation='relu')(lstm)
    dropped = Dropout(0.35)(densed)
    output = Dense(units=7, activation='softmax')(dropped)

    rnn_model = Model(sequence_input, output)
    rnn_model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["acc"])
    return rnn_model


# 训练模型
if __name__ == '__main__':
    fun_list = [create_mlp, create_cnn, create_rnn]
    name_list = ['mlp_model.h5', 'cnn_model.h5', 'rnn_model.h5']
    choice = input("choose a network you want.(0 for mlp, 1 for cnn, 2 for rnn):")
    choice = int(choice)
    model_path = os.path.join('..', 'model', name_list[choice])
    word_index, embeddings_matrix = get_word2vec_dictionaries()
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        x_train, y_train = load_file('train', word_index)
        model = fun_list[choice](embeddings_matrix)
        x_valid, y_valid = load_file('valid', word_index)
        model.fit(x_train, y_train, batch_size=100, epochs=13, validation_data=(x_valid, y_valid))

    x_test, y_test = load_file('test', word_index)
    # x_test = tokenizer(xxx, word_index)
    # y_test = np.array(yyy)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

    model.save(model_path)
