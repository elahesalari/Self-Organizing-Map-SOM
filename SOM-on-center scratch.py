import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    return data


def on_center(data, k, start_time):
    x = data
    epoch = 100
    weight = np.random.uniform(0, 1, size=(k, k, data.shape[1]))
    eta0 = 0.1
    sigma0 = 1
    batch_size = 64
    re_weight = weight.reshape((k * k, -1))
    for n in range(epoch):
        index = random.sample(range(0, x.shape[0]), x.shape[0])
        for i in range(round(x.shape[0] / batch_size)):
            idx_bt = index[i * batch_size:(i + 1) * batch_size]
            y_in = x[idx_bt] @ re_weight.T
            # continue
            winner = np.argmax(y_in, axis=1)

            sigma = sigma0 * np.exp(-n / epoch)
            eta = eta0 * np.exp(-n / epoch)
            batch = len(idx_bt)

            h_k, w_winner = gaussian_neighborhood(winner, k, sigma, weight, batch)
            h_k = h_k.reshape(batch, k * k).T

            delta_w = eta * h_k @ (x[idx_bt] - w_winner)
            weight += delta_w.reshape((k, k, -1)) / batch

            print(f'\r epoch:{n} , batch:{i} , time:{time.time() - start_time}', end='')

        if eta < 0.01:
            break

    print('\n -----------*****-----------')
    calc_distance(x, weight, k)


def calc_distance(x, weights, k):
    re_weight = weights.reshape((k * k, -1))
    y = x @ re_weight.T
    win_neuron = np.argmax(y, axis=1)
    _, count = np.unique(win_neuron, return_counts=True)

    matrix = count.reshape(k, k)
    print(f'Count of Data in each neuron in {k}x{k} topologies: \n', matrix)

    row = win_neuron // k
    column = win_neuron % k
    w_win = weights[row, column]

    dis = np.sqrt(np.sum((x - w_win) ** 2, axis=1))
    sum_dis = np.sum(dis)

    print('Sum of the nearlablity of each data to winning neuron: ', sum_dis)

    plot_hits(matrix, k)


def gaussian_neighborhood(winner, k, sigma, weight, batch):
    h_k = np.zeros((batch, k, k))
    row = winner // k
    column = winner % k
    r_k = np.array([row, column]).T  # position of winning neuron
    w_winner = weight[row, column]

    for i in range(k):
        for j in range(k):
            d_jk = np.linalg.norm(r_k - np.array([i, j]), axis=1)
            h_k[:, i, j] = np.exp(- d_jk ** 2 / (2 * sigma ** 2))

    return h_k, w_winner


def _neuron(data):
    K = [3, 4, 5]
    for k in K:
        start_time = time.time()
        on_center(data, k, start_time)
        end_time = time.time()
        print(f'Time for topology {k} :{end_time - start_time} \n')
        print('--------------------------------------------')


def plot_hits(matrix, k):
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, cmap='BuGn')
    for i in range(k):
        for j in range(k):
            txt = ax.text(j, i, matrix[i, j], ha='center', va='center', color='k')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = load_data()
    _neuron(data)
