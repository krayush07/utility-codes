from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import random


# Label is the last column in a tab separated file

def create_kfold_train_test(input_file, seed=None):
    file = open(input_file)
    file_arr = file.readlines()
    if seed is not None:
        random.seed(seed)
    random.shuffle(file_arr)

    labels = []

    for each_line in file_arr:
        labels.append(each_line.strip().split('\t')[-1])

    label_encoder = preprocessing.LabelEncoder()

    label_encoder.fit(labels)
    label_id = label_encoder.transform(labels)
    split = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
    data_split = split.split(file_arr, label_id)

    train_file = open(input_file + '_TRAIN_SPLIT.txt', 'w')
    valid_file = open(input_file + '_VALID_SPLIT.txt', 'w')

    for train_idx, valid_idx in data_split:
        for each_train_idx in train_idx:
            train_file.write(file_arr[each_train_idx].strip() + '\n')

        for each_valid_idx in valid_idx:
            valid_file.write(file_arr[each_valid_idx].strip() + '\n')
        break

    train_file.close()
    valid_file.close()


def main():
    create_kfold_train_test('input_file_path', seed=4)


if __name__ == '__main__':
    main()
