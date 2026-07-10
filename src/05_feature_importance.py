import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_sleep_health.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "models" / "model_results.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "models" / "confusion_matrix.csv"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Sleep_Disorder"

LIGHT_PURPLE = "#CDB4DB"
DARK_PURPLE = "#9D4EDD"


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    X = pd.get_dummies(X, drop_first=True)

    return X


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    results = pd.read_csv(RESULTS_PATH)
    confusion = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=results,
        x="Model",
        y="Accuracy",
        color=LIGHT_PURPLE
    )
    plt.title("Porównanie dokładności modeli")
    plt.xlabel("Model")
    plt.ylabel("Dokładność")
    plt.ylim(0, 1)
    save_plot("07_model_comparison.png")

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Purples"
    )
    plt.title("Macierz pomyłek najlepszego modelu")
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Rzeczywista klasa")
    save_plot("08_confusion_matrix.png")

    X = prepare_data(df)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Selected model does not provide feature_importances_.")

    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("Top 10 most important features:")
    print(importances.head(10))

    top_features = importances.head(10).copy()

    feature_labels = {
        "Systolic_BP": "Ciśnienie skurczowe",
        "Diastolic_BP": "Ciśnienie rozkurczowe",
        "BMI_Category_Overweight": "BMI: nadwaga",
        "BMI_Category_Normal Weight": "BMI: waga prawidłowa",
        "Daily_Steps": "Liczba kroków dziennie",
        "Age": "Wiek",
        "Heart_Rate": "Tętno",
        "Stress_Level": "Poziom stresu",
        "Quality_of_Sleep": "Jakość snu",
        "Sleep_Duration": "Czas snu",
        "Physical_Activity_Level": "Poziom aktywności fizycznej"
    }

    top_features["Feature_Label"] = top_features["Feature"].map(feature_labels)
    top_features["Feature_Label"] = top_features["Feature_Label"].fillna(top_features["Feature"])

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature_Label",
        color=LIGHT_PURPLE
    )
    plt.title("Najważniejsze cechy modelu")
    plt.xlabel("Ważność cechy")
    plt.ylabel("")
    save_plot("09_feature_importance.png")
