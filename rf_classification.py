import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
import time
import numpy as np
import os

def rf_classification(
    train_csv, test_csv, feature,
    n_estimators_list=[10, 30, 50, 100, 150, 200],
    random_state=42,
    class_order=['unripe', 'ripe', 'overripe'],
    save_csv=True,
    out_csv='rf_estimators_comparison_metrics.csv'
):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=['filename', 'label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['filename', 'label'])
    y_test = test_df['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for n_estimators in n_estimators_list:
        print(f"\n--- Random Forest (n_estimators={n_estimators}) on {feature} ---")
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test_scaled)
        acc = np.mean(y_pred == y_test) * 100
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        test_loss = 100 - acc

        results.append({
            'n_estimators': n_estimators,
            'Accuracy (%)': acc,
            'Precision (%)': prec,
            'Recall (%)': rec,
            'F1-score (%)': f1,
            'Test Loss (%)': test_loss,
            'Training Time (s)': training_time
        })

        print(classification_report(y_test, y_pred, labels=class_order, digits=4))
        print(f"Training time: {training_time:.4f} seconds")

        cmatrix = confusion_matrix(y_test, y_pred, labels=class_order)
        disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=class_order)
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f"Confusion Matrix (RF, n={n_estimators}) on {feature}")
        plt.show()


        y_test_bin = label_binarize(y_test, classes=class_order)
        y_proba_raw = model.predict_proba(X_test_scaled)
        model_classes = list(model.classes_)
        y_proba = np.zeros_like(y_proba_raw)
        for idx, label in enumerate(class_order):
            label_index = model_classes.index(label)
            y_proba[:, idx] = y_proba_raw[:, label_index]

        if y_test_bin.shape[1] == 1:
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba[:, 1])
            auc_score = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='orange', label=f"{class_order[1]} (area = {auc_score:.2f})")
            plt.plot([0, 1], [0, 1], 'k--', label="Random (area = 0.5)")
            plt.title(f"ROC Curve (Binary, RF n={n_estimators}) on {feature}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
        else:
            n_classes = y_test_bin.shape[1]
            color_dict = {
                'unripe': 'navy',      
                'partial_ripe': 'orange', 
                'ripe': 'green',
                'overripe': 'red'    
            }
            plt.figure(figsize=(8, 6))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                class_name = class_order[i]
                curve_color = color_dict.get(class_name, 'black')
                plt.plot(
                    fpr, tpr,
                    color=curve_color,
                    label=f"{class_name} (area = {auc_score:.2f})",
                    linewidth=2
                )
            plt.plot([0, 1], [0, 1], 'k--', label="Random (area = 0.5)")
            plt.title(f"Multiclass ROC Curve (RF, n={n_estimators}) on {feature}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['n_estimators'], results_df['Accuracy (%)'], marker='o', label='Accuracy')
    plt.xlabel("Number of Estimators (Trees)")
    plt.ylabel("Score (%)")
    plt.title(f"Random Forest - Metrics vs n_estimators ({feature})")
    plt.ylim([results_df['Accuracy (%)'].min()-2, 100])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.plot(results_df['n_estimators'], results_df['Test Loss (%)'], marker='o', color='red', label='Test Loss')
    plt.xlabel("Number of Estimators (Trees)")
    plt.ylabel("Test Loss (%)")
    plt.title(f"Random Forest - Test Loss vs n_estimators ({feature})")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results_df
