import matplotlib.pyplot as plt
import numpy as np


# plot accuracy vs class, num of instances vs class etc.
def plot_bar(yval, xval, xlabel, ylabel, title):
    index = np.arange(len(xval))
    plt.bar(index, yval)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(index, xval, fontsize=7, rotation=30)
    plt.title(title)
    plt.show()


# compare accuracy of two methods vs class
def plot_multiple_bar(y1val, y2val, xval, xlabel, ylabel, title):
    ax = plt.subplot(111)
    bar_width = 0.3
    x = np.arange(20)
    y1_bar = ax.bar(x - bar_width, y1val, width=bar_width, color='g', align='center')
    y2_bar = ax.bar(x, y2val, width=bar_width, color='r', align='center')
    # ax.bar(x + bar_width, k, width=bar_width, color='r', align='center')
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    index = np.arange(len(xval))
    plt.xticks(index, xval, fontsize=7, rotation=50)
    plt.title(title)
    plt.legend([y1_bar, y2_bar], ['Y1 Value', 'Y2 Value'])
    plt.show()
