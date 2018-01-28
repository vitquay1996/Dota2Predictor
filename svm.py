
"""
Train dota 2 match-predictor
"""

import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def main():
    """Orchestrate the retrival of data, training and testing."""
    print("getting data")
    data = get_data()

    # Get classifier
    from sklearn.svm import SVC
    # clf = SVC(probability=False,  # cache_size=200,
    #           kernel="rbf", C=2.8, gamma=.0073)
    clf = RandomForestClassifier(n_estimators=128, max_features=3)
    print("Start fitting. This may take a while")

    # take all of it - make that number lower for experiments
    examples = len(data['train']['X'])
    clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])

    analyze(clf, data)

    # Save model in a file
    joblib.dump(clf, 'predictor_0.4.pkl')

def analyze(clf, data):
    """
    Analyze how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """
    # Get confusion matrix
    from sklearn import metrics
    predicted = clf.predict(data['test']['X'])
    print("predicted is", predicted, data['test']['y'])
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'],
                                                     predicted))

def get_data():
    """
    Get data ready to learn with.

    Returns
    -------
    dict
    """
    from sklearn.datasets import fetch_mldata
    from sklearn.utils import shuffle
    # mnist = fetch_mldata('MNIST original')
    y = np.array([[]])
    x = np.array([])
    # Get data from json file
    with open('match_result.json') as json_result:
        results = json.load(json_result)
        num_results = 0
        y = np.array([int(a) for a in results[:1942]])

        with open('match_detail.json') as json_details:
            details = json.load(json_details)
            num_details = 0
            x = np.array([[(float(item) -50)/50 for item in row] for row in details[:1942]])

            # x = mnist.data
            # y = mnist.target
            # print(x.shape, y.shape)
            # Scale data to [-1, 1] - This is of mayor importance!!!

            x, y = shuffle(x, y, random_state=0)

            from sklearn.cross_validation import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.1,
                                                                random_state=42)
            data = {'train': {'X': x_train,
                            'y': y_train},
                    'test': {'X': x_test,
                            'y': y_test}}
    # print(data)
    return data


if __name__ == '__main__':
    main()