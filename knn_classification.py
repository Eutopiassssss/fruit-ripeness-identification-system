import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import time

def split_per_class(csv_path, train_output, test_output, feature, test_size=0.2, random_state=42,shuffle=True):
    df = pd.read_csv(csv_path)
    train_list = []
    test_list = []
    
    for class_label in df['label'].unique():
        class_df = df[df['label'] == class_label]
        train_df, test_df = train_test_split(
            class_df, test_size=test_size, random_state=random_state, shuffle=True, stratify=None
        )
        train_list.append(train_df)
        test_list.append(test_df)
        
    final_train_df = pd.concat(train_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    final_train_df.to_csv(train_output, index=False)
    final_test_df = pd.concat(test_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    final_test_df.to_csv(test_output, index=False)
    print(f" Train set for {feature} saved in {train_output} ({len(final_train_df)} samples)")
    print(f" Test set for {feature} saved in {test_output} ({len(final_test_df)} samples)")


def knn_classification(
    train_csv, test_csv, feature,
    k_values=[1, 3, 5, 7, 10],
    class_order=['unripe', 'ripe', 'overripe'],
    save_csv=True,
    out_csv='knn_k_comparison_metrics.csv'
):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=['filename', 'label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['filename', 'label'])
    y_test = test_df['label']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    for k in k_values:
        print(f"\n--- KNN (k={k}) on {feature} Feature ---")
        start_time = time.time()
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test) * 100
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        test_loss = 100 - acc

        results.append({
            'k': k,
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
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (k={k}) --- {feature}")
        plt.show()

        if hasattr(model, "predict_proba"):
            y_proba_raw = model.predict_proba(X_test)
            proba_df = pd.DataFrame(y_proba_raw, columns=model.classes_)
            y_proba = proba_df.reindex(columns=class_order, fill_value=0).to_numpy()
            y_test_bin = label_binarize(y_test, classes=class_order)
            if y_test_bin.shape[1] == 1:
                fpr, tpr, _ = roc_curve(y_test_bin, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, color='orange', label=f"{class_order[1]} (area = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], 'k--', label='Random (area = 0.5)')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve (Binary) (k={k}) --- {feature}")
                plt.legend(loc='lower right')
                plt.grid()
                plt.show()
            else:
                n_classes = y_test_bin.shape[1]
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                color_dict = {
                    'unripe': 'navy',
                    'partial_ripe': 'orange',
                    'ripe': 'green',
                    'overripe': 'red'
                }
                plt.figure()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                for i in range(n_classes):
                    class_name = class_order[i]
                    curve_color = color_dict.get(class_name, 'black')
                    plt.plot(
                        fpr[i], tpr[i],
                        color=curve_color,
                        label=f"{class_name} (area = {roc_auc[i]:.2f})"
                    )
                plt.plot([0, 1], [0, 1], 'k--', label='Random (area = 0.5)')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f'Multiclass ROC Curve (k={k}) --- {feature}')
                plt.legend(loc='lower right')
                plt.grid()
                plt.show()
        else:
            print("KNN model does not support predict_proba, skipped ROC")

    results_df = pd.DataFrame(results)

    if save_csv:
        results_df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['k'], results_df['Accuracy (%)'], marker='o', label='Accuracy')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Score (%)")
    plt.title(f"KNN - Metrics vs k ({feature})")
    plt.ylim([results_df['Accuracy (%)'].min()-2, 100])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(results_df['k'], results_df['Test Loss (%)'], marker='o', color='red', label='Test Loss')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Test Loss (%)")
    plt.title(f"KNN - Test Loss vs k ({feature})")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results_df
