import random
from typing import List

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

#QDA: 82,09 %
#LDA: 81,78 %

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


def classify(point, liked_rv, disliked_rv) -> bool:
    if (2450/10_000) * liked_rv.pdf(point) > (7550/10_000) * disliked_rv.pdf(point):
        return 1
    else:
        return 0


def generate_QDA_rvs(training_data):
    global t
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
    global t
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


def test_methods_cv(dataset, no_tests, num_folds, method):
    results = []
    
    class0 = [x for x in dataset if x[-1] == 0]
    class1 = [x for x in dataset if x[-1] == 1]

    for _ in range(no_tests):

        # Split the data into k folds
        random.shuffle(class0)
        random.shuffle(class1)
        folds0 = split_into_folds(class0, num_folds)
        folds1 = split_into_folds(class1, num_folds)
        folds = [a + b for a, b in zip(folds0, folds1)]

        # Train on k-1 folds and test on the last
        fold_results = []
        for j in range(num_folds):
            testing = folds[j]
            training = sum((x for x in folds if x!=testing), [])

            if method == "QDA":
                liked_rv, disliked_rv = generate_QDA_rvs(training)
            elif method == "LDA":
                liked_rv, disliked_rv = generate_LDA_rvs(training)
            else:
                raise ValueError("Method not supported")
        
            correct = 0
            for point in testing:
                ans = classify(point[:-1], liked_rv, disliked_rv)
                if ans == point[-1]:
                    correct += 1
            
            fold_results.append(100*(correct/len(testing)))

        results.append(np.mean(fold_results))

    return np.mean(results)


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


def main():

    dataset = get_dataset("../train_test_2025.csv")


    ts = []
    anss = []
    for i in np.linspace(0, 0.01, 10):
        global t
        t = i
        print(t)
        ans = test_methods_cv(dataset, no_tests=10, num_folds=5, method="LDA")
        ts.append(i)
        anss.append(ans)
    
    plt.plot(ts, anss)
    plt.xlabel("t")
    plt.ylabel("accuracy")
    plt.show()
   
    ans = test_methods_cv(dataset, no_tests=10, num_folds=5, method="LDA")
    print("LDA accuracy:", ans, "%")


if __name__ == "__main__":
    main()