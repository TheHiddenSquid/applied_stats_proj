import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

#QDA: 82,86 %
#LDA: 83,13 %

# ROC <10min
# QDA-AOC: 0.890
# LDA-AOC: 0.892

def get_dataset(filename: str) -> List[List[float]]:
    
    df = pd.read_csv(filename)

    to_remove = ["fnlwgt","education"]
    for val in to_remove:
        df.drop(val, axis=1, inplace=True)

    threshold = 30
    mask = df.groupby("native-country")["native-country"].transform("size") < threshold
    df.loc[mask, "native-country"] = "Other"
    
    to_one_hot_encode = ["workclass","marital-status","occupation","relationship","race","sex", "native-country", "over50k"]
    df = pd.get_dummies(df, columns=to_one_hot_encode, drop_first=True, dtype=int)
    
    to_normalize = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    for var in to_normalize:
        tmp = df[var]
        df[var] = (tmp - tmp.mean()) / tmp.std()

    list_of_data = df.values.tolist()

    return list_of_data


def classify(point, liked_rv, disliked_rv, cutoff = 1) -> bool:
    if  liked_rv.logpdf(point) - disliked_rv.logpdf(point) > cutoff * np.log(7550/2450):
        return 1
    else:
        return 0
    



def generate_QDA_rvs(training_data):
    t = 0.1277 # Optimal from cv (acc)
    t = 0.154 # Optimal from cv (auc)
    liked_songs_array = np.concatenate([np.array(song[:-1], ndmin=2).transpose() for song in training_data if song[-1] == 1], axis=1)
    liked_mean = np.mean(liked_songs_array, axis=1)
    liked_cov = np.cov(liked_songs_array)
    liked_cov = (1-t) * liked_cov + t * np.eye(liked_cov.shape[0])
    liked_rv = stats.multivariate_normal(liked_mean, liked_cov, allow_singular=True)

    disliked_songs_array = np.concatenate([np.array(song[:-1], ndmin=2).transpose() for song in training_data if song[-1] == 0], axis=1)
    disliked_mean = np.mean(disliked_songs_array, axis=1)
    disliked_cov = np.cov(disliked_songs_array)
    disliked_cov = (1-t) * disliked_cov + t * np.eye(disliked_cov.shape[0])
    disliked_rv = stats.multivariate_normal(disliked_mean, disliked_cov, allow_singular=True)

    return liked_rv, disliked_rv


def generate_LDA_rvs(training_data):
    t = 0.001 # Optimal from cv (acc)
    t = 0.043 # Optimal from cv (auc)
    songs_array = np.concatenate([np.array(song[:-1], ndmin=2).transpose() for song in training_data], axis=1)
    cov = np.cov(songs_array)
    cov = (1-t) * cov + t * np.eye(cov.shape[0])

    liked_songs_array = np.concatenate([np.array(song[:-1], ndmin=2).transpose() for song in training_data if song[-1] == 1], axis=1)
    liked_mean = np.mean(liked_songs_array, axis=1)
    liked_rv = stats.multivariate_normal(liked_mean, cov, allow_singular=True)

    disliked_songs_array = np.concatenate([np.array(song[:-1], ndmin=2).transpose() for song in training_data if song[-1] == 0], axis=1)
    disliked_mean = np.mean(disliked_songs_array, axis=1)
    disliked_rv = stats.multivariate_normal(disliked_mean, cov, allow_singular=True)

    return liked_rv, disliked_rv


def test_methods_cv(dataset, num_tests, num_folds, method, cutoff):
    results = np.zeros((num_tests, 5))
    
    class0 = [x for x in dataset if x[-1] == 0]
    class1 = [x for x in dataset if x[-1] == 1]

    for i in range(num_tests):

        # Split the data into k folds
        random.shuffle(class0)
        random.shuffle(class1)
        folds0 = split_into_folds(class0, num_folds)
        folds1 = split_into_folds(class1, num_folds)
        folds = [a + b for a, b in zip(folds0, folds1)]

        # Train on k-1 folds and test on the last
        fold_results = np.zeros((num_folds, 5))
        for j in range(num_folds):
            testing = folds[j]
            training = sum((x for x in folds if x!=testing), [])

            if method == "QDA":
                liked_rv, disliked_rv = generate_QDA_rvs(training)
            elif method == "LDA":
                liked_rv, disliked_rv = generate_LDA_rvs(training)
            else:
                raise ValueError("Method not supported")
        
            CM = np.zeros((2,2))
            for point in testing:
                guess = classify(point[:-1], liked_rv, disliked_rv, cutoff)
                correct = point[-1]
                guess = 0 if guess == 1 else 1
                correct = 0 if correct == 1 else 1
                CM[correct, guess] += 1
     
            accuracy = (CM[0,0]+CM[1,1])/np.sum(CM)
            recall = CM[0,0]/(CM[0,0]+CM[0,1])
            fall_out = CM[1,0]/(CM[1,0]+CM[1,1])
            precision = CM[0,0]/(CM[0,0]+CM[1,0])
            F1 = 2*recall*precision / (recall+precision)
            
            fold_results[j,:] = accuracy, recall, fall_out, precision, F1 

        results[i,:] = np.mean(fold_results, axis=0)

    return np.mean(results, axis=0)


def split_into_folds(dataset, num_folds):
    n = len(dataset)
    fold_len = int(np.floor(n / num_folds))
    num_bigger_folds = n - num_folds * fold_len
    
    folds = []
    start = 0
    end = fold_len

    for _ in range(num_bigger_folds):
        folds.append(dataset[start: end+1])
        start = end+1
        end += fold_len+1

    while end <= n:
        folds.append(dataset[start: end])
        start = end
        end += fold_len

    return folds


def ROC(dataset, num_tests = 5):
    pairs_LDA = []
    pairs_QDA = []

    for cutoff in np.linspace(-3, 3, 50):
        ans = test_methods_cv(dataset, num_tests=num_tests, num_folds=5, method="LDA", cutoff=cutoff)
        pairs_LDA.append((ans[2],ans[1]))
        ans = test_methods_cv(dataset, num_tests=num_tests, num_folds=5, method="QDA", cutoff=cutoff)
        pairs_QDA.append((ans[2],ans[1]))

    for cutoff in np.linspace(-20, 20, 20):
        ans = test_methods_cv(dataset, num_tests=num_tests, num_folds=5, method="QDA", cutoff=cutoff)
        pairs_QDA.append((ans[2],ans[1]))

    pairs_LDA.append((0,0))
    pairs_LDA.append((1,1))
    pairs_QDA.append((0,0))
    pairs_QDA.append((1,1))

    pairs_LDA.sort(key=lambda x:x[0])
    pairs_QDA.sort(key=lambda x:x[0])

    AUC_LDA = np.trapz([x[1] for x in pairs_LDA], [x[0] for x in pairs_LDA])
    AUC_QDA = np.trapz([x[1] for x in pairs_QDA], [x[0] for x in pairs_QDA])

    return pairs_LDA, pairs_QDA, AUC_LDA, AUC_QDA
    

def main():

    dataset = get_dataset("../train_test_2025.csv")

    pairs_LDA, pairs_QDA, AUC_LDA, AUC_QDA = ROC(dataset, num_tests=5)

    print("AOC LDA:", AUC_LDA)
    print("AOC QDA:", AUC_QDA)

    plt.plot([x[0] for x in pairs_LDA], [x[1] for x in pairs_LDA], label="LDA")
    plt.plot([x[0] for x in pairs_QDA], [x[1] for x in pairs_QDA], label="QDA")
    plt.plot([0,1], [0,1], color="gray", label="_")
    plt.xlabel("Fall-Out")
    plt.ylabel("Recall")
    plt.title(f"ROC Curve")
    plt.legend(loc="lower right")
    plt.grid("true")
    plt.margins(0)
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.01)
    plt.savefig("ROC.eps", format="eps")
    plt.show()


if __name__ == "__main__":
    main()