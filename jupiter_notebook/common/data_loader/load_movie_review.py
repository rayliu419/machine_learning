from nltk.corpus import movie_reviews
import pandas as pd
import sys
sys.path.append("..")
from common.utils.nlp_utils import *


def load_movie_review():
    """
    每个文件是一个评论!
    :return:
    data frame
    """
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((movie_reviews.words(fileid), category))
    movie_text = []
    movie_lable = []
    for doc in documents:
        movie_text.append(doc[0])
        movie_lable.append(doc[1])
    df = pd.DataFrame()
    df['text'] = movie_text
    df['label'] = movie_lable
    return df


def load_movie_review_raw():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            raw_text = movie_reviews.raw(fileid)
            merge_text = merge_to_single_line(raw_text)
            documents.append((merge_text, category))
    movie_text = []
    movie_lable = []
    for doc in documents:
        movie_text.append(doc[0])
        movie_lable.append(doc[1])
    df = pd.DataFrame()
    df['text'] = movie_text
    df['label'] = movie_lable
    return df


if __name__ == "__main__":
    df = load_movie_review()
    print(df['text'][0])
    print(df['label'][0])
    df.info()

    df2 = load_movie_review_raw()
    print(df2['text'][0])
    print(df2['label'][0])
