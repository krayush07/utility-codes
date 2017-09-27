import random


def create_random_train_test(input_file, split_ratio=0.7, seed=None):
    file = open(input_file)
    file_arr = file.readlines()
    if seed is not None:
        random.seed(seed)
    random.shuffle(file_arr)

    train_file = open(input_file + '_TRAIN_SPLIT1.txt', 'w')
    valid_file = open(input_file + '_VALID_SPLIT1.txt', 'w')

    num_line = int(split_ratio * len(file_arr))

    for i in range(num_line):
        train_file.write(file_arr[i].strip() + '\n')

    for i in range(num_line, len(file_arr), 1):
        valid_file.write(file_arr[i].strip() + '\n')

    train_file.close()
    valid_file.close()


def main():
    create_random_train_test('input_file_path', split_ratio=0.5)


if __name__ == '__main__':
    main()
