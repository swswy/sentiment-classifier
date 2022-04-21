import re


def getSecIndex(sen):
    count = 0
    fir_id = 0
    sec_id = 0
    for i in range(0, len(sen)):
        if sen[i] == ',':
            count += 1
            if count == 1:
                fir_id = i
            elif count == 2:
                sec_id = i
                break
    return fir_id, sec_id


filter_regex = re.compile(r"[^\w ']+")
text_file = open("../data/isear_test.csv", 'r', encoding='utf-8')
text = ''
result = ''
emotion_list = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
try:
    emotion = ''
    for line in text_file:
        fir, sec = getSecIndex(line)
        emotion = line[fir + 1:sec]
        result += str(emotion_list.index(emotion))
        result += '\n'
        quotation_sen = line[sec + 1:len(line) - 1]
        single_text = filter_regex.sub('', quotation_sen)
        single_text = single_text.strip()
        single_text = single_text.lower()
        text += single_text
        text += '\n'
except:
    print("the error name is " + emotion)
text_file.close()
new_text_file = open("../pretreatment/x_test", 'w', encoding='utf-8')
new_text_file.write(text)
new_text_file.close()
new_result_file = open("../pretreatment/y_test", "w", encoding='utf-8')
new_result_file.write(result)
new_result_file.close()
