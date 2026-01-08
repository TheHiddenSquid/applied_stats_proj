import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def knn_classify(point, A_feat, A_class, A_sq) -> bool:
    
    b = np.array(point)
    v_sq = np.dot(b, b) 
    squared_l2 = A_sq + v_sq - 2 * (A_feat @ b)
    
    global k
    indices_of_k_nearest = np.argpartition(squared_l2, k)[:k]
    ones = A_class[indices_of_k_nearest].sum()

    frac = ones / k
    threshold = 0.5

    if frac >= threshold:
        return 1
    else:
        return 0


def test_methods_cv(dataset, no_tests, num_folds):
    results = np.zeros((no_tests, 5))
    
    class0 = [x for x in dataset if x[-1] == 0]
    class1 = [x for x in dataset if x[-1] == 1]

    for i in range(no_tests):

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

            # Specific for KNN
            A_training = np.array(training)
            A_feat = A_training[:,:-1]
            A_class = A_training[:,-1]
            A_sq = np.einsum('ij,ij->i', A_feat, A_feat)
        
            CM = np.zeros((2,2))
            for point in testing:
                guess = knn_classify(point[:-1], A_feat, A_class, A_sq)
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


def main():

    dataset = get_dataset("../train_test_2025.csv")

    ks = []
    accs = []
    for i in range(3,101,2):
        global k
        k = i
        print(i)
        ans = test_methods_cv(dataset, no_tests=5, num_folds=5)
        ks.append(k)
        accs.append(ans[0])

    plt.plot(ks, accs)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("Parameter tuning")
    plt.show()
    



if __name__ == "__main__":
    main()