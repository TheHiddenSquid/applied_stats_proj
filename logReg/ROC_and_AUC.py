import numpy as np
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import logistic_regression as lr
import data_loader as dl 

def auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined with no positive or no negative samples")

    count = 0.0

    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5

    return count / (n_pos * n_neg)



def main():
    X_train, X_test, y_train, y_test = dl.load_and_split_data(
        "train_test_2025.csv", test_size=0.2, random_state=11
    )

    model = lr.LogisticRegression(learning_rate=0.4, n_iters=1000, l2_lambda=10.0)
    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)
    accuracy = np.mean(model_predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plt.figure(figsize=(7, 5))
    y_proba = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    # Save FPR and TPR to text file
    """
    with open("ROC_plots.txt", "a") as f:
        f.write("\n")
        f.write("\n")
        f.write("log_reg_fpr:")
        f.write(fpr.tolist().__str__())
        f.write("\n")
        f.write("\n")
        f.write("log_reg_tpr:")
        f.write(tpr.tolist().__str__())"""

    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
    plt.legend(loc='lower right')
    plt.show()

    metric_auc = auc_score(y_test, y_proba)
    print(f"AUC Score: {metric_auc:.4f}")

if __name__ == "__main__":
    main()