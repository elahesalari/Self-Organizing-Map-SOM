import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from minisom import MiniSom
from matplotlib.gridspec import GridSpec
import time
from pylab import plot, axis, show, pcolor, colorbar, bone


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

    return data, label


def on_center(data, label, k):
    som = MiniSom(x=k, y=k, input_len=data.shape[1], sigma=1.0, learning_rate=0.2, neighborhood_function='gaussian',
                  topology='rectangular', activation_distance='euclidean')
    som.random_weights_init(data)

    print('training...')
    som.train_random(data, num_iteration=1000)
    print('...ready!')

    winners = []
    matrix = np.zeros((k, k))
    category_dic = {}

    for i, x in enumerate(data):
        win = som.winner(x)
        winners.append(win)
        matrix[win[0], win[1]] += 1

        if (win[0], win[1]) not in category_dic.keys():
            category_dic[(win[0], win[1])] = []
        category_dic[(win[0], win[1])].append(label[i])

    plot_category(category_dic, k)

    print(f'SOM hits for {k}x{k} topologies: \n', matrix)
    plot_hits(matrix, k)

    # Euclidean distance
    weight = som.get_weights()
    w_win = np.zeros((data.shape))
    for j in range(data.shape[0]):
        w_win[j] = weight[winners[j][0], winners[j][1]]

    dis = np.linalg.norm(data - w_win, axis=1)
    sum_dis = np.sum(dis)

    print('Sum of the nearlablity of each data to winning neuron: ', sum_dis)


def plot_hits(matrix, k):
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, cmap='PuBu')
    for i in range(k):
        for j in range(k):
            txt = ax.text(j, i, matrix[i, j], ha='center', va='center', color='k')
    fig.tight_layout()
    ax.set_title(f'SOM Hits Plot {k}x{k}')
    plt.show()


def plot_category(category_dic, k):
    marker = ['♚', '✣', '♠', '♣', '★']
    colors = ['red', 'blue', 'orangered', 'lime', 'yellow']

    m_matrix = np.zeros((k, k), int)
    for idx, d in enumerate(category_dic.keys()):
        val, count = np.unique(category_dic[d], return_counts=True)
        argmax_idx = np.argmax(count)
        majority = val[argmax_idx]

        m_matrix[d[0], d[1]] = majority
        # print(f'val:{val} , count:{count} ')
    print('similarity matrix for show categories:\n', m_matrix)

    fig, ax = plt.subplots()
    image = ax.imshow(m_matrix, cmap='Blues')
    for i in range(k):
        for j in range(k):
            txt = ax.text(j, i, marker[m_matrix[i, j]], ha='center', va='center', color=colors[m_matrix[i, j]],
                          fontsize=20)
    fig.tight_layout()
    ax.set_title(f'SOM Nearlably Category Plot {k}x{k}')
    show()


def _neuron(data, y):
    K = [3, 4, 5]
    for k in K:
        start_time = time.time()
        on_center(data, y, k)
        end_time = time.time()
        print(f'Time for topology {k} :{end_time - start_time} \n')
        print('--------------------------------------------')


if __name__ == '__main__':
    data, y = load_data()
    _neuron(data, y)
