from tensorflow.keras.preprocessing.text import Tokenizer
import hanlp

cn_word_tokenizer = hanlp.load('CTB6_CONVSEG')


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
    """
    单字分词
    :param sentence:
    :return:
    """
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


def seg_chinese_word_single(sentence):
    """
    中文词语分词，不是单字
    :param sentence:
    :return:
    """
    return cn_word_tokenizer(sentence)


def seg_chinese_word_array(sentence_array):
    return cn_word_tokenizer(sentence_array)


def encode_chinese_word_to_int(seg_sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seg_sentences)
    encoded = tokenizer.texts_to_sequences(seg_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, encoded, vocab_size


def get_word_dict(seg_sentences):
    """
    对于某些情况，需要在外面转成int
    :param seg_sentences:
    :return:
    """
    tokenizer = Tokenizer()
    # tokenizer会把0预留出来作为表示句子结束<EOS>，如果还要加入<BOS>怎么做？
    tokenizer.fit_on_texts(seg_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def encode_chinese_word_to_int_with_tok(tok, seg_sentences):
    return tok.texts_to_sequences(seg_sentences)


if __name__ == "__main__":
    print(hanlp.pretrained.ALL)
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

    print("==================test chinese word seg=========================")
    seg_word_senteces = seg_chinese_word_array(data)
    print(seg_word_senteces)
    tokenizer, encoded, vocab_size = encode_chinese_word_to_int(seg_word_senteces)
    for word, index in tokenizer.word_index.items():
        print("{} - {}".format(word, index))
    print(encoded)
    print("==================test chinese word seg=========================")

