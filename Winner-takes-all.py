import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    data = np.load('feature_vector.npy')

    df = pd.read_csv('bbc-text.csv')
    df.loc[df['category'] == 'business', 'category'] = 0
    df.loc[df['category'] == 'entertainment', 'category'] = 1
    df.loc[df['category'] == 'politics', 'category'] = 2
    df.loc[df['category'] == 'sport', 'category'] = 3
    df.loc[df['category'] == 'tech', 'category'] = 4
    label = list(df.iloc[:, 0])
    _, cn = np.unique(label, return_counts=True)
    print('count of data in each class:', cn)

    return data, label


def winner_takes_all(data, c):
    x = data
    alpha = 0.03
    epoch = 100
    weight = np.random.uniform(0, 1, size=(x.shape[1], c))
    batch_size = 32
    for ep in range(epoch):
        index = random.sample(list(np.arange(0, x.shape[0])), x.shape[0])
        for i in range(round(x.shape[0] / batch_size)):
            idx_bt = index[i * batch_size:(i + 1) * batch_size]
            y = x[idx_bt].dot(weight)
            winner = np.argmax(y, axis=1)
            for j in range(len(winner)):
                delta_w = alpha * (x[idx_bt[j]] - weight[:, winner[j]])
                weight[:, winner[j]] += delta_w

            print(f'\repoch: {ep} , batch: {i}', end='')
    # calculate number of data in each cluster
    k = x @ weight
    win = np.argmax(k, axis=1)
    dt_cluster, count = np.unique(win, return_counts=True)
    print('\ncount of data in each cluster:', count)

    return win, count


def calc_confusion_matrix(win, x, label, c):
    win = win.tolist()
    confusion_matrix = np.zeros((c, c))
    for n in range(c):
        index = [label[i] for i, e in enumerate(win) if e == n]
        val, count = np.unique(index, return_counts=True)
        n_max = np.argmax(count)
        row = val[n_max]

        confusion_matrix[row, val] = count
    print('confusion matrix : \n', confusion_matrix)

    return confusion_matrix


def plot_hits(matrix, c, count_list):
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title('Histogram Winner')
    label_name = ['business', 'entertainment', 'politics', 'sport', 'tech']
    r = np.arange(c)
    ax[0].bar(r, height=count_list, color='lightblue')
    ax[0].set_xticks(r)
    ax[0].set_xticklabels(label_name)

    image = ax[1].imshow(matrix, cmap='Pastel2')
    ax[1].set_title('Confusion matrix')
    for i in range(c):
        for j in range(c):
            txt = ax[1].text(j, i, matrix[i, j], ha='center', va='center', color='k')
    ax[1].set_xticks(r)
    ax[1].set_xticklabels(label_name)
    ax[1].set_yticks(r)
    ax[1].set_yticklabels(label_name)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    c = 5
    data, label = load_data()
    win, count_list = winner_takes_all(data, c)
    conf = calc_confusion_matrix(win, data, label, c)
    plot_hits(conf, c, count_list)
