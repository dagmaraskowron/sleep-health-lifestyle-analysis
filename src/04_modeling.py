import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_sleep_health.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
RESULTS_PATH = MODELS_DIR / "model_results.csv"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.csv"

TARGET_COLUMN = "Sleep_Disorder"


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X, y = prepare_data(df)

    print("Features shape:")
    print(X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42))
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42
        )
    }

    results = []
    best_model = None
    best_model_name = None
    best_accuracy = 0
    best_predictions = None
    best_classes = None

    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 50)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_name == "Random Forest":
            joblib.dump(model, MODELS_DIR / "random_forest_model.joblib")

        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": model_name,
            "Accuracy": accuracy
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
            best_predictions = y_pred
            best_classes = model.classes_

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    confusion = confusion_matrix(
        y_test,
        best_predictions,
        labels=best_classes
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=best_classes,
        columns=best_classes
    )

    confusion_df.to_csv(CONFUSION_MATRIX_PATH)

    joblib.dump(best_model, MODEL_PATH)

    print("\nModel comparison:")
    print(results_df)

    print("\nBest model:")
    print(best_model_name)
