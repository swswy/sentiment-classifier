import os.path
from hw3 import load_file, get_word2vec_dictionaries
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import f1_score
import numpy as np
import json

if __name__ == "__main__":

    word_index, embeddings_matrix = get_word2vec_dictionaries()
    text = json.dumps(word_index)
    with open("../data/dict.json", 'w', encoding="utf-8") as f:
        f.write(text)

    x_test, y_test = load_file('test', word_index)
    name_list = ['mlp_model.h5', 'cnn_model.h5', 'rnn_model.h5']
    choice = input("choose a network you want.(0 for mlp, 1 for cnn, 2 for rnn):")
    choice = int(choice)
    model_path = os.path.join('..', 'model', name_list[choice])
    model = load_model(model_path)
    scores = model.evaluate(x_test, y_test)
    print('the performance of ' + name_list[choice] + ' is:')
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

    y_pred = model.predict(x_test)

    # f-score
    y_label_true = np.argmax(y_test, axis=1)
    y_label_pred = np.argmax(y_pred, axis=1)
    print("f1 score:(macro)")
    print(f1_score(y_label_true, y_label_pred, average='macro'))

    print("f1 score:(micro)")
    print(f1_score(y_label_true, y_label_pred, average='micro'))

    #plot_model(model, to_file='../'+name_list[choice][:-3]+'.png',show_shapes=True)
