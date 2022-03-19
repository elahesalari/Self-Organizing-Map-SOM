import pandas as pd
import numpy as np
import re


def read_text():
    df = pd.read_csv('bbc-text.csv')
    news = df.iloc[:, -1]
    stopw = stopword(news)
    w = non_letter(stopw)
    text = shortword(w)
    df.iloc[:, -1] = text

    return df


def stopword(news):
    f = open("stopwords.txt", "r")
    stword = f.read().split()
    for j in range(0, news.shape[0]):
        resultwords = [word for word in news[j].split() if word.lower() not in stword]
        news[j] = ' '.join(resultwords)

    return news


def non_letter(news):
    for i in range(news.shape[0]):
        txt = re.sub('[^A-Za-z]', ' ', news[i])
        news[i] = txt

    return news


def shortword(news):
    for i in range(news.shape[0]):
        old_str = ' '.join(re.findall("[a-zA-Z]+", news[i]))
        news[i] = ' '.join([w for w in old_str.split() if len(w) >= 3])

    return news


def document_preprocessing(data):
    N = data.shape[0]
    data = np.array(data.iloc[:, :])

    words = []
    for w in range(N):
        words += data[w, -1].split()

    words = np.unique(words).tolist()

    feature = len(words)

    tf = np.zeros((N, feature))
    for i in range(N):
        doc = data[i, -1].split()

        unique_doc, count_term = np.unique(doc, return_counts=True)
        number_of_words_in_d = len(unique_doc)

        for j in range(number_of_words_in_d):
            term = unique_doc[j]
            index = words.index(term)
            tf[i, index] = count_term[j] / len(doc)  # count of term`j in doc / to number of whole word in doc

        print(f'\rcalc tf matrix : {i} of {N}', end='')

    df = np.zeros((1, feature))
    for w in range(feature):
        for i in range(N):
            if words[w] in data[i, -1]:
                df[0, w] += 1
        print(f'\rclac df matrix : {w} of {feature}', end='')

    tf_idf = np.log(1 + tf) * np.log(N / df)

    # save list in file
    np.save('feature_vector.npy', tf_idf)


if __name__ == '__main__':
    text = read_text()
    document_preprocessing(text)
