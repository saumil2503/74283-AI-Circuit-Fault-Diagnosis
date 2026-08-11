import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

DATASET_FOLDER = "dataset"
MODEL_FOLDER = "models"
OUTPUT_FOLDER = "outputs"

FEATURES = [
    "i1", "i2", "i3", "i4", "i5",
    "i6", "i7", "i8", "i9",
    "o1", "o2", "o3", "o4", "o5"
]

FILES = {
    "z2": "74283_modified_z2_nand2_to_and2_ground_truth_0noise.csv",
    "z17": "74283_modified_z17_and5_to_nand5_ground_truth_0noise.csv",
    "z18": "74283_modified_z18_and2_to_nand2_ground_truth_0noise.csv",
    "o1": "74283_modified_o1_nor5_to_or5_ground_truth_0noise.csv"
}


# --------------------------------------------------
# 2. CREATE REQUIRED FOLDERS
# --------------------------------------------------

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# 3. LOAD AND COMBINE DATASETS
# --------------------------------------------------

print("=" * 60)
print("  74283 DIGITAL CIRCUIT ANOMALY CLASSIFICATION")
print("=" * 60)

dataframes = []

for anomaly, filename in FILES.items():

    path = os.path.join(
        DATASET_FOLDER,
        filename
    )

    if not os.path.exists(path):

        print(f"\nERROR: Dataset file not found:")
        print(path)

        exit()

    df = pd.read_csv(path)

    # Add target/class label
    df["anomaly_class"] = anomaly

    dataframes.append(df)

    print(
        f"Loaded {anomaly:4s}: "
        f"{len(df)} samples"
    )


data = pd.concat(
    dataframes,
    ignore_index=True
)


# Keep only ML features and target
data = data[
    FEATURES + ["anomaly_class"]
]


print("\nDataset successfully combined.")

print(
    f"Total samples : {len(data)}"
)

print(
    f"Features      : {len(FEATURES)}"
)

print(
    f"Classes       : "
    f"{data['anomaly_class'].nunique()}"
)


print("\nClass distribution:")

print(
    data["anomaly_class"].value_counts()
)


# --------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# --------------------------------------------------

X = data[FEATURES]

y = data[
    "anomaly_class"
]


# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# --------------------------------------------------
# SAVE HELD-OUT TEST SET
# --------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

held_out_test = X_test.copy()

held_out_test["anomaly_class"] = y_test.values

test_set_path = os.path.join(
    OUTPUT_FOLDER,
    "held_out_test_set.csv"
)

held_out_test.to_csv(
    test_set_path,
    index=False
)

print(
    f"Held-out test set saved to: "
    f"{test_set_path}"
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)


print(
    f"\nTraining samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)


print(
    "\nTraining percentage : "
    f"{len(X_train) / len(data) * 100:.2f}%"
)

print(
    "Testing percentage  : "
    f"{len(X_test) / len(data) * 100:.2f}%"
)


# --------------------------------------------------
# 6. SAVE HELD-OUT TEST DATASET
# --------------------------------------------------

# Recombine X_test with its true labels.
#
# This file contains ONLY samples that were NOT
# used to train the machine-learning models.

test_data = X_test.copy()

test_data[
    "anomaly_class"
] = y_test


# Reset row numbers for cleaner CSV
test_data = test_data.reset_index(
    drop=True
)


test_data_path = os.path.join(
    OUTPUT_FOLDER,
    "test_data.csv"
)


test_data.to_csv(
    test_data_path,
    index=False
)


print(
    f"\nHeld-out test dataset saved to:"
)

print(
    test_data_path
)

print(
    f"Saved test observations: "
    f"{len(test_data)}"
)


# --------------------------------------------------
# 7. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

    ]),


    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),


    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),


    "SVM (RBF)": Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        )

    ])
}


# --------------------------------------------------
# 8. TRAIN AND COMPARE MODELS
# --------------------------------------------------

results = {}

trained_models = {}


print("\n" + "=" * 60)
print("TRAINING ML MODELS")
print("=" * 60)


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results[name] = accuracy

    trained_models[name] = model


    print(
        f"{name:22s}: "
        f"{accuracy * 100:.2f}%"
    )


# --------------------------------------------------
# 9. SHOW BEST PERFORMING MODEL
# --------------------------------------------------

best_model_name = max(
    results,
    key=results.get
)


best_accuracy = results[
    best_model_name
]


print("\n" + "-" * 60)

print(
    f"Best performing model : "
    f"{best_model_name}"
)

print(
    f"Best accuracy         : "
    f"{best_accuracy * 100:.2f}%"
)


# --------------------------------------------------
# 10. SELECT SVM AS PROJECT MODEL
# --------------------------------------------------

# We currently use the SVM classifier in predict.py.
#
# The model comparison is still reported separately,
# so we can transparently compare all algorithms.

final_model = trained_models[
    "SVM (RBF)"
]


final_predictions = final_model.predict(
    X_test
)


final_accuracy = accuracy_score(
    y_test,
    final_predictions
)


print("\n" + "=" * 60)
print("FINAL PROJECT MODEL: SVM (RBF)")
print("=" * 60)


print(
    f"\nAccuracy: "
    f"{final_accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")


report = classification_report(
    y_test,
    final_predictions,
    digits=4
)


print(report)


# --------------------------------------------------
# 11. SAVE CLASSIFICATION REPORT
# --------------------------------------------------

report_path = os.path.join(
    OUTPUT_FOLDER,
    "svm_classification_report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        "74283 DIGITAL CIRCUIT "
        "ANOMALY CLASSIFICATION\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        "Final Model: SVM (RBF)\n"
    )

    file.write(
        f"Accuracy: "
        f"{final_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        report
    )


print(
    f"\nClassification report saved to:"
)

print(
    report_path
)


# --------------------------------------------------
# 12. SAVE TRAINED SVM MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_FOLDER,
    "svm_anomaly_classifier.pkl"
)


joblib.dump(
    final_model,
    model_path
)


print(
    f"\nSVM model saved to:"
)

print(
    model_path
)


# --------------------------------------------------
# 13. SAVE MODEL COMPARISON TABLE
# --------------------------------------------------

comparison_df = pd.DataFrame({

    "Model": list(
        results.keys()
    ),

    "Accuracy": [
        value * 100
        for value in results.values()
    ]

})


comparison_csv_path = os.path.join(
    OUTPUT_FOLDER,
    "model_comparison.csv"
)


comparison_df.to_csv(
    comparison_csv_path,
    index=False
)


print(
    f"\nModel comparison table saved to:"
)

print(
    comparison_csv_path
)


# --------------------------------------------------
# 14. SAVE MODEL COMPARISON GRAPH
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    results.keys(),
    [
        value * 100
        for value in results.values()
    ]
)


plt.title(
    "ML Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()


comparison_path = os.path.join(
    OUTPUT_FOLDER,
    "model_comparison.png"
)


plt.savefig(
    comparison_path,
    dpi=200
)


plt.close()


print(
    f"Model comparison graph saved to:"
)

print(
    comparison_path
)


# --------------------------------------------------
# 15. SAVE CONFUSION MATRIX
# --------------------------------------------------

labels = [
    "o1",
    "z17",
    "z18",
    "z2"
]


cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=labels
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


display.plot()


plt.title(
    "SVM Confusion Matrix"
)


plt.tight_layout()


confusion_path = os.path.join(
    OUTPUT_FOLDER,
    "svm_confusion_matrix.png"
)


plt.savefig(
    confusion_path,
    dpi=200
)


plt.close()


print(
    f"Confusion matrix saved to:"
)

print(
    confusion_path
)


# --------------------------------------------------
# 16. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)


print(
    f"""
Total dataset samples : {len(data)}
Training samples      : {len(X_train)}
Held-out test samples : {len(X_test)}

Final project model   : SVM (RBF)
SVM test accuracy     : {final_accuracy * 100:.2f}%

Best compared model   : {best_model_name}
Best model accuracy   : {best_accuracy * 100:.2f}%

Saved files:

models/
  svm_anomaly_classifier.pkl

outputs/
  test_data.csv
  svm_classification_report.txt
  model_comparison.csv
  model_comparison.png
  svm_confusion_matrix.png
"""
)