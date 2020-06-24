from tensorflow.keras.preprocessing.text import Tokenizer


def load_chinese_and_seg(file, line_num=100000):
    """
    # 将中文用空格分开并返回，作为encode_to_int的输入
    :param file:
    :param line_num:
    :return:
    """
    sentence_array = []
    line_index = 0
    with open(file) as infile:
        for line in infile:
            line_index += 1
            line = line.strip()
            seg_str = seg_chinese_single(line)
            sentence_array.append(seg_str)
            if line_index == line_num:
                break
    return sentence_array


def seg_chinese_array(data):
    sentence_array = []
    for sentence in data:
        seg_str = seg_chinese_single(sentence)
        sentence_array.append(seg_str)
    return sentence_array


def seg_chinese_single(sentence):
    current = []
    for i in sentence:
        if i != "-":
            current.append(i)
    seg_str = " ".join(current)
    return seg_str


def encode_chinese_to_int(seg_sentences):
    """
    将空格分开的中文encode成int，作为神经网络的输入
    :param chinese_poi_name:
    :return:
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seg_sentences)
    encoded = tokenizer.texts_to_sequences(seg_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, encoded, vocab_size


if __name__ == "__main__":
    print("==================test seg chinese to int======================")
    data = [
        "建功北里",
        "西直门"
    ]
    seg_sentences = seg_chinese_array(data)
    print(seg_sentences)
    tokenizer, encoded, vocab_size = encode_chinese_to_int(seg_sentences)
    for word, index in tokenizer.word_index.items():
        print("{} - {}".format(word, index))
    print(encoded)
    print("==================test seg chinese to int======================")