from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_extraction.text import CountVectorizer


def train(train_filename, test_filename):
    train_file = open(train_filename, 'r')
    output_file = open(test_filename + '_nn_output', 'w')
    test_file = open(test_filename, 'r')

    train_utt, train_label = [], []
    test_utt, test_label = [], []

    for line in train_file:
        utt, label = line.strip().split('\t')
        train_utt.append(utt)
        train_label.append(label)

    for line in test_file:
        utt, label = line.strip().split('\t')
        test_utt.append(utt)
        test_label.append(label)

    vectorizer = CountVectorizer(stop_words='english')
    train_features = vectorizer.fit_transform([utt for utt in train_utt])
    test_features = vectorizer.transform([utt for utt in test_utt])

    nn = NearestCentroid()
    nn.fit(train_features, [int(labels) for labels in train_label])

    predictions = nn.predict(test_features)

    for pred in predictions:
        output_file.write(str(pred) + '\n')

    train_file.close()
    test_file.close()
    output_file.close()

    # fpr, tpr, thresholds = metrics.roc_curve(np.asarray(test_label), predictions, pos_label=1)
    # print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))


root_data_dir = 'path_to_root_dir'
train(root_data_dir + 'path_to_trainset', root_data_dir + 'path_to_testset')
