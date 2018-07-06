from sklearn.datasets import load_iris
from sklearn import tree

def load_data_set():
    iris = load_iris()
    return iris


def train_model(iris):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)
    return clf


def display_image(clf):
    tree.export_graphviz(clf, out_file='dtree.dot',
                    filled=True, rounded=True,
                    special_characters=True)

if __name__ == '__main__':
    iris_data = load_iris()
    decision_tree_classifier = train_model(iris_data)
    display_image(clf=decision_tree_classifier)