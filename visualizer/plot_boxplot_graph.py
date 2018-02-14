import pandas as pd
import matplotlib.pyplot as plt


# to plot class wise probability range/error
def plot_boxplot(input_filename):
    df = pd.read_csv(input_filename, sep="\t")
    df.boxplot(column='prob_score', by='gold_label', rot=30, return_type='axes', medianprops={'linestyle': '-', 'linewidth': 5})
    plt.show()
