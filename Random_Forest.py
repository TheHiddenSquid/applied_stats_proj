import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, RocCurveDisplay, roc_curve

def get_dataset(normalize=True):
    data = pd.read_csv("train_test_2025.csv", encoding = 'cp1252')

    data = data.drop(["fnlwgt","education"], axis=1)
    one_hot_columns = ["workclass","marital-status","occupation","relationship","race","sex", "native-country"]
    data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True, dtype=int)
    data['‚™'] = data['‚™'].map({'yes': 1, 'no': 0})
    
    if normalize:
        normalize_columns = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        for var in normalize_columns:
            tmp = data[var]
            data[var] = (tmp - tmp.mean()) / tmp.std()

    return data

def param_test(dataset):
    X = dataset.drop("‚™", axis=1)
    y = dataset["‚™"]
    
    param_grid = {
    "n_estimators": [60, 65, 70, 80, 90, 100],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
    )
    
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.get_params())
    print("Best CV score:", grid_search.best_score_)
    
    return grid_search


def cv_forest(dataset, no_tests=5, cv=5):
    results = np.zeros((no_tests, 4))
    for i in range(no_tests):
        dataset = dataset.sample(frac=1)
        
        X = dataset.drop("‚™", axis=1)
        y = dataset["‚™"]
        
        rforest = RandomForestClassifier(n_estimators=75, criterion="gini", max_depth=17, min_samples_leaf=4)
        fold_acc = np.mean(cross_validate(rforest, X, y, cv=cv)["test_score"])
        
        test_predictions = cross_val_predict(rforest, X, y, cv=cv)

        rec = recall_score(y, test_predictions)
        prec = precision_score(y, test_predictions)
        f1 = f1_score(y, test_predictions)
        
        results[i] = fold_acc, rec, prec, f1
    return np.mean(results, axis=0)


def ROC_curve(dataset, test_size=0.2):
    X = dataset.drop("‚™", axis=1)
    y = dataset["‚™"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    rforest = RandomForestClassifier(n_estimators=65, criterion = "gini", max_depth=17, min_samples_leaf=4)
    rforest.fit(X_train, y_train)
    
    RocCurveDisplay.from_estimator(rforest, X_test, y_test)
    plt.show()
    
    y_pred = rforest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, threshold = roc_curve(y_test, rforest.predict_proba(X_test)[:,1])
    return acc, f1, fpr, tpr, threshold
    
    
data = get_dataset()

#estimator = param_test(data)
#print(estimator.decision_function(X))


print("Folding test metrics: accuracy, recall, precision, F1")
print(cv_forest(data))


acc, f1, fpr, tpr, threshold = ROC_curve(data)
print("Random test accuracy:")
print(acc)
print("Random test F1-Score:")
print(f1)
print("Random test fpr-values:")
print(fpr.tolist())
print("Random test tpr-values:")
print(tpr.tolist())
print("Random test thresholds:")
print(threshold)