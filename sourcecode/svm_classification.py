import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

def svm_classification(
    train_csv: str,
    test_csv: str,
    feature_name: str,
    degrees: list,
    strategies: list,
    kernels: list,
    class_order: list,
    output_dir: str = "results", 
    save_csv: bool = True,
    out_csv_name: str = 'svm_results.csv'
) -> (pd.DataFrame, pd.DataFrame):

    os.makedirs(output_dir, exist_ok=True)
    out_csv      = os.path.join(output_dir, out_csv_name)
    pairwise_csv = out_csv.replace('.csv', '_pairwise.csv')

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    X_train, y_train = (
        train_df.drop(columns=['filename','label']),
        train_df['label']
    )
    X_test, y_test = (
        test_df.drop(columns=['filename','label']),
        test_df['label']
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    multi_results   = []
    pairwise_results= []

    for kernel in kernels:
        for strat in strategies:
            if strat not in ('ovo','ovr'):
                raise ValueError("strategies must be 'ovo' or 'ovr'")
            for deg in degrees:
                print(f"\n=== SVM (kernel={kernel}, degree={deg}, strat={strat.upper()}) ===")
                t0 = time.time()
                svc   = SVC(kernel=kernel, degree=deg, probability=False, random_state=42)
                model = OneVsOneClassifier(svc) if strat=='ovo' else OneVsRestClassifier(svc)
                model.fit(X_train, y_train)
                elapsed = time.time() - t0

                y_pred = model.predict(X_test)
                acc    = np.mean(y_pred==y_test)*100
                prec   = precision_score(y_test, y_pred, average='weighted', zero_division=0)*100
                rec    = recall_score(y_test, y_pred, average='weighted', zero_division=0)*100
                f1_    = f1_score(y_test, y_pred, average='weighted', zero_division=0)*100
                loss   = 100.0 - acc

                multi_results.append({
                    'Kernel':kernel, 'Strategy':strat.upper(), 'Degree':deg,
                    'Accuracy (%)':acc, 'Precision (%)':prec,
                    'Recall (%)':rec, 'F1-score (%)':f1_,
                    'Test Loss (%)':loss, 'Train Time (s)':elapsed
                })

                print(classification_report(y_test, y_pred, labels=class_order, digits=4))
                cm   = confusion_matrix(y_test, y_pred, labels=class_order)
                disp = ConfusionMatrixDisplay(cm, display_labels=class_order)
                disp.plot(cmap=plt.cm.Oranges)
                plt.title(f"Confusion Matrix --- {feature_name} (kernel={kernel}, strat={strat.upper()}, deg={deg})")
                plt.show()

                if strat=='ovo':
                    for c1, c2 in combinations(class_order, 2):
                        mask_train = y_train.isin([c1, c2])
                        mask_test  = y_test.isin([c1, c2])
                        if not (mask_train.any() and mask_test.any()): continue
                        bsvc = SVC(kernel=kernel, degree=deg, probability=False, random_state=42)
                        bsvc.fit(X_train[mask_train], y_train[mask_train])
                        ap = np.mean(bsvc.predict(X_test[mask_test])==y_test[mask_test])*100
                        pairwise_results.append({
                            'Kernel':kernel, 'Strategy':'OVO', 'Degree':deg,
                            'PairType':'Class vs Class', 'Class1':c1, 'Class2':c2,
                            'Accuracy (%)':ap
                        })
                else:
                    for cls in class_order:
                        true_bin = (y_test == cls).astype(int)
                        pred_bin = (y_pred == cls).astype(int)
                        ap = np.mean(true_bin == pred_bin)*100
                        pairwise_results.append({
                            'Kernel':kernel, 'Strategy':'OVR', 'Degree':deg,
                            'PairType':'Class vs Rest', 'Class1':cls, 'Class2':'(Rest)',
                            'Accuracy (%)':ap
                        })

    df_multi    = pd.DataFrame(multi_results)
    df_pairwise = pd.DataFrame(pairwise_results)

    if save_csv:
        df_multi.to_csv(out_csv, index=False)
        df_pairwise.to_csv(pairwise_csv, index=False)
        print(f"\nSaved:\n  {out_csv}\n  {pairwise_csv}\n")

    return df_multi, df_pairwise
