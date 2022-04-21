import json
import re
import numpy as np
import os.path
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

filter_regex = re.compile(r"[^\w ']+")
emotion_list = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
name_list = ['mlp_model.h5', 'cnn_model.h5', 'rnn_model.h5']
choice = input("choose a network you want.(0 for mlp, 1 for cnn, 2 for rnn):")
choice = int(choice)
model_path = os.path.join('..', 'model', name_list[choice])
model = load_model(model_path)
text = ''
with open("../data/dict.json", 'r', encoding="utf-8") as f:
    text = f.read()
dict_text = json.loads(text)

while True:
    text_str = input("the sentence is: ")
    if text_str == '':
        break
    text_str = text_str.replace('á', 'a').replace('-', ' ').lower()
    text_str = filter_regex.sub('', text_str)

    cw = list(filter(None, text_str.split(' ')))
    word_id = []
    # 把词转换为编号
    for word in cw:
        try:
            temp = dict_text[word]
            word_id.append(dict_text[word])
        except:
            word_id.append(0)
    word_id = np.array(word_id)
    word_id = word_id[np.newaxis, :]
    sequences = pad_sequences(word_id, maxlen=120, padding='post')
    result = np.argmax(model.predict(sequences))
    print(emotion_list[int(result)])
