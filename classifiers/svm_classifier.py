from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score


def train(train_filename, test_filename):
    train_file = open(train_filename, 'r')
    output_file = open(test_filename + '_svm_output', 'w')
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

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    train_features = vectorizer.fit_transform([utt for utt in train_utt])
    test_features = vectorizer.transform([utt for utt in test_utt])

    svmc = svm.SVC(kernel='linear', C=1, probability=True)

    scores = cross_val_score(svmc, train_features, train_label, cv=3)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    svmc.fit(train_features, [str(labels) for labels in train_label])

    predictions = svmc.predict(test_features)
    probabilities = svmc.predict_proba(test_features)

    # for pred, prob in zip(predictions, probabilities):
    #     pred_idx = 0 if pred == -1 else 1
    #     # print(prob[pred_idx]), pred, prob
    #     if (prob[pred_idx] >= 0.98):
    #         if(pred_idx == 0):
    #             output_file.write('-1' + '\n')
    #         else:
    #             output_file.write('1' + '\n')
    #     else:
    #         output_file.write('0' + '\n')

    test_acc = 0.0
    total_correct = 0.0

    for idx, pred in enumerate(predictions):
        if (pred == test_label[idx]):
            total_correct += 1
        else:
            print(test_utt[idx], pred, test_label[idx])
        output_file.write(str(pred) + '\n')

    print(total_correct / len(test_label))
    train_file.close()
    test_file.close()
    output_file.close()

    # fpr, tpr, thresholds = metrics.roc_curve(np.asarray(test_label), predictions, pos_label=1)
    # print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))


root_data_dir = 'path_to_root_dir'
train(root_data_dir + 'path_to_trainset', root_data_dir + 'path_to_testset')
