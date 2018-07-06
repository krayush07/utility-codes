import matplotlib.pyplot as plt
import numpy

'''
    Plot the histogram for predicted probability values.
    Instance example: PREDICTED LABEL tab PREDICTED PROBABILITY tab TEST LABEl
'''

def plot_histograms(input_array):
    plt.hist(input_array, bins=10)
    plt.xlabel('Probabilities')
    plt.ylabel('Count')
    plt.show()

def plot_histogram_util(input_file):
    lines = open(input_file, 'r').readlines()
    prob_arr = []
    for line in lines:
        line_split = line.strip().split('\t')
        pred_label = line_split[0]
        test_label = line_split[2]
        if pred_label == test_label:
            prob_arr.append(float(line.strip().split('\t')[1]))

    plot_histograms(prob_arr)

plot_histogram_util('FILEPATH')


