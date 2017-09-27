import numpy as np
import matplotlib.pyplot as pp
import itertools


def plot_cosine_matrix(cm, title, vmax, vmin):
    pp.imshow(cm, interpolation='nearest', vmax=vmax, vmin=vmin)
    pp.title(title)
    pp.colorbar()

    thresh = 0.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pp.text('', '', '', horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    pp.tight_layout()

    pp.figure()
    pp.show()


def get_cosine_similarity(str1, str2, delimiter):
    arr1 = np.array(map(np.float32, str1.split(delimiter)), dtype=np.float32)
    arr2 = np.array(map(np.float32, str2.split(delimiter)), dtype=np.float32)

    arr_size = len(arr1)

    match_count = 0.0

    for i in range(arr_size):
        if (arr1[i] == arr2[i]):
            match_count += 1

    cosine_val = match_count / arr_size

    return cosine_val


def get_cosine_similarity_one_vs_all(inputfile, output_filename_suffix, delimiter):
    with open(inputfile) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    output = open(output_path + '/cosine_similarity_one_vs_all_' + output_filename_suffix + '.csv', 'w')

    num_lines = len(content)

    cosine_matrix = np.zeros(shape=[num_lines, num_lines])

    for i in range(num_lines):
        output_str = ''
        print 'Processing line %s ' % str(i)
        for j in range(num_lines):
            if (j < i):
                output_str += str(cosine_matrix[j][i]) + '\t'
                cosine_matrix[i][j] = cosine_matrix[j][i]
            elif (i == j):
                output_str += '1.0\t'
            else:
                cosine_val = get_cosine_similarity(content[i], content[j], delimiter)
                output_str += str(cosine_val) + '\t'
                cosine_matrix[i][j] = cosine_val
        output.write(output_str.strip() + '\n')
    output.close()


output_path = 'output_path'
get_cosine_similarity_one_vs_all(inputfile='input_path', output_filename_suffix='BOOLEAN_COSINE_ONE_VS_ALL', delimiter=' ')
plot_cosine_matrix(cm=np.genfromtxt('input_path'), title='cosine', vmin=0, vmax=1)
