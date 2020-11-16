"""
better english process utils than english_preprocess
"""

import nltk
import pandas as pd
pd.set_option('max_colwidth', 200)
import re
import string
string.punctuation


def remove_punctuations(text):
    """
    移除所有的标点符号
    :param text: string
    :return:
    """
    text_no_punctuations = "".join([char for char in text if char not in string.punctuation])
    return text_no_punctuations


def remove_punctuations_array(string_array):
    text_no_punctuations = []
    for s in string_array:
        if s not in string.punctuation:
            text_no_punctuations.append(s)
    return text_no_punctuations


def tokenize(text):
    """
     #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    :param text:
    :return:
    """
    tokens = re.split('\W+', text)
    return tokens


def remove_stopwords(tokenized_list):
    # All English Stopwords
    stop_word = nltk.corpus.stopwords.words('english')
    # To remove all stopwords
    text = [word for word in tokenized_list if word not in stop_word]
    return text


def stemming(tokenized_text):
    """
    过去分词变为原型
    :param tokenized_text:
    :return:
    """
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in tokenized_text]
    return text


def lemmatizing(tokenized_text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text


if __name__ == "__main__":
    print(remove_punctuations("the big cat jump to floor!"))
    print(tokenize("the big cat jump to floor!"))
    print(remove_stopwords(['the', 'big', 'cat', 'jump', 'to', 'floor', '']))
    print(stemming(['the', 'big', 'cat', 'jumped', 'to', 'floor', '']))
    print(lemmatizing(['the', 'big', 'cat', 'jumped', 'to', 'floor', '']))