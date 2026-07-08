import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/sleep_health_lifestyle.csv"
IMAGES_DIR = "images"

SLEEP_DISORDER_PALETTE = {
    "Brak": "#4C78A8",
    "Bezdech senny": "#F58518",
    "Bezsenność": "#54A24B",
}

BINARY_PALETTE = {
    "Nie": "#4C78A8",
    "Tak": "#E45756",
}

MAIN_COLOR = "#4C78A8"


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook")

    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titleweight"] = "bold"


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")

    sleep_disorder_translation = {
        "None": "Brak",
        "Sleep Apnea": "Bezdech senny",
        "Insomnia": "Bezsenność",
    }

    df["sleep_disorder_pl"] = df["sleep_disorder"].map(sleep_disorder_translation)

    df["has_sleep_disorder"] = df["sleep_disorder"].apply(
        lambda value: "Nie" if value == "None" else "Tak"
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[20, 30, 40, 50, 60],
        labels=["20-30", "31-40", "41-50", "51-60"],
        include_lowest=True,
    )

    df["sleep_duration_group"] = pd.cut(
        df["sleep_duration"],
        bins=[0, 6, 8, 12],
        labels=["Krótki sen", "Zalecany zakres snu", "Długi sen"],
    )

    return df


def print_dataset_overview(df):
    print("\nPierwsze wiersze danych:")
    print(df.head())

    print("\nRozmiar zbioru danych:")
    print(df.shape)

    print("\nNazwy kolumn:")
    print(df.columns)

    print("\nBrakujące wartości:")
    print(df.isnull().sum())

    print("\nRozkład zaburzeń snu:")
    print(df["sleep_disorder_pl"].value_counts())

    print("\nRozkład zmiennej binarnej: czy występuje zaburzenie snu?")
    print(df["has_sleep_disorder"].value_counts())

    print("\nRozkład grup wiekowych:")
    print(df["age_group"].value_counts())


def print_grouped_statistics(df):
    print("\nŚrednia jakość snu według zaburzenia snu:")
    print(
        df.groupby("sleep_disorder_pl")["quality_of_sleep"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nŚrednia długość snu według zaburzenia snu:")
    print(
        df.groupby("sleep_disorder_pl")["sleep_duration"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nŚredni poziom stresu według zaburzenia snu:")
    print(
        df.groupby("sleep_disorder_pl")["stress_level"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nŚrednia jakość snu według kategorii BMI:")
    print(
        df.groupby("bmi_category")["quality_of_sleep"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nŚredni poziom stresu według kategorii BMI:")
    print(
        df.groupby("bmi_category")["stress_level"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nŚrednia jakość snu według poziomu aktywności fizycznej:")
    print(
        df.groupby("physical_activity_level")["quality_of_sleep"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nKorelacja między aktywnością fizyczną a jakością snu:")
    print(df[["physical_activity_level", "quality_of_sleep"]].corr())


def get_correlation_matrix(df):
    numeric_columns = [
        "age",
        "sleep_duration",
        "quality_of_sleep",
        "physical_activity_level",
        "stress_level",
        "heart_rate",
        "daily_steps",
    ]

    correlation_matrix = df[numeric_columns].corr()

    print("\nMacierz korelacji:")
    print(correlation_matrix)

    return correlation_matrix


def save_plot(file_name):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, file_name), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def add_bar_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fontsize=9, padding=3)


def create_visualizations(df, correlation_matrix):

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="quality_of_sleep",
        color=MAIN_COLOR,
        edgecolor="black",
    )
    add_bar_labels(ax)
    plt.title("Rozkład jakości snu")
    plt.xlabel("Jakość snu")
    plt.ylabel("Liczba osób")
    save_plot("sleep_quality_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="sleep_duration",
        y="quality_of_sleep",
        hue="sleep_disorder_pl",
        palette=SLEEP_DISORDER_PALETTE,
        s=80,
        alpha=0.8,
    )
    plt.title("Długość snu a jakość snu")
    plt.xlabel("Długość snu w godzinach")
    plt.ylabel("Jakość snu")
    plt.legend(title="Zaburzenie snu")
    save_plot("sleep_duration_vs_quality.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="stress_level",
        y="quality_of_sleep",
        hue="sleep_disorder_pl",
        palette=SLEEP_DISORDER_PALETTE,
        s=80,
        alpha=0.8,
    )
    plt.title("Poziom stresu a jakość snu")
    plt.xlabel("Poziom stresu")
    plt.ylabel("Jakość snu")
    plt.legend(title="Zaburzenie snu")
    save_plot("stress_vs_sleep_quality.png")

    bmi_order = ["Normal", "Normal Weight", "Overweight", "Obese"]
    existing_bmi_order = [
        category for category in bmi_order if category in df["bmi_category"].unique()
    ]

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="bmi_category",
        y="quality_of_sleep",
        order=existing_bmi_order,
        hue="bmi_category",
        palette="Set2",
        legend=False,
    )
    plt.title("Jakość snu według kategorii BMI")
    plt.xlabel("Kategoria BMI")
    plt.ylabel("Jakość snu")
    save_plot("sleep_quality_by_bmi.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="physical_activity_level",
        y="quality_of_sleep",
        hue="sleep_disorder_pl",
        palette=SLEEP_DISORDER_PALETTE,
        s=80,
        alpha=0.8,
    )
    plt.title("Aktywność fizyczna a jakość snu")
    plt.xlabel("Poziom aktywności fizycznej")
    plt.ylabel("Jakość snu")
    plt.legend(title="Zaburzenie snu")
    save_plot("physical_activity_vs_sleep_quality.png")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="RdBu_r",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Macierz korelacji zmiennych zdrowotnych i stylu życia")
    save_plot("correlation_matrix.png")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="sleep_disorder_pl",
        order=df["sleep_disorder_pl"].value_counts().index,
        hue="sleep_disorder_pl",
        palette=SLEEP_DISORDER_PALETTE,
        edgecolor="black",
        legend=False,
    )
    add_bar_labels(ax)
    plt.title("Rozkład zaburzeń snu")
    plt.xlabel("Zaburzenie snu")
    plt.ylabel("Liczba osób")
    save_plot("sleep_disorder_distribution.png")

    plt.figure(figsize=(9, 5))
    ax = sns.countplot(
        data=df,
        x="bmi_category",
        hue="has_sleep_disorder",
        order=existing_bmi_order,
        palette=BINARY_PALETTE,
        edgecolor="black",
    )
    add_bar_labels(ax)
    plt.title("Występowanie zaburzeń snu według kategorii BMI")
    plt.xlabel("Kategoria BMI")
    plt.ylabel("Liczba osób")
    plt.legend(title="Zaburzenie snu")
    save_plot("sleep_disorder_by_bmi.png")

    top_occupations = df["occupation"].value_counts().head(8).index
    occupation_df = df[df["occupation"].isin(top_occupations)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=occupation_df,
        x="occupation",
        y="quality_of_sleep",
        hue="occupation",
        palette="Set3",
        legend=False,
    )
    plt.title("Jakość snu według zawodu")
    plt.xlabel("Zawód")
    plt.ylabel("Jakość snu")
    plt.xticks(rotation=30, ha="right")
    save_plot("sleep_quality_by_occupation.png")

    occupation_stress = (
        df.groupby("occupation")["stress_level"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=occupation_stress.values,
        y=occupation_stress.index,
        color="#F58518",
        edgecolor="black",
    )
    plt.title("Top 10 zawodów według średniego poziomu stresu")
    plt.xlabel("Średni poziom stresu")
    plt.ylabel("Zawód")
    save_plot("stress_level_by_occupation.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="heart_rate",
        y="stress_level",
        hue="has_sleep_disorder",
        palette=BINARY_PALETTE,
        s=80,
        alpha=0.8,
    )
    plt.title("Tętno a poziom stresu")
    plt.xlabel("Tętno")
    plt.ylabel("Poziom stresu")
    plt.legend(title="Zaburzenie snu")
    save_plot("heart_rate_vs_stress.png")


def prepare_ml_data(df):
    features = [
        "gender",
        "age",
        "occupation",
        "sleep_duration",
        "quality_of_sleep",
        "physical_activity_level",
        "stress_level",
        "bmi_category",
        "heart_rate",
        "daily_steps",
    ]

    target = "has_sleep_disorder"

    X = df[features]
    y = df[target]

    numeric_features = [
        "age",
        "sleep_duration",
        "quality_of_sleep",
        "physical_activity_level",
        "stress_level",
        "heart_rate",
        "daily_steps",
    ]

    categorical_features = [
        "gender",
        "occupation",
        "bmi_category",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return X, y, preprocessor, numeric_features, categorical_features


def create_confusion_matrix_plot(y_test, y_pred, model_name, file_name):
    matrix = confusion_matrix(y_test, y_pred, labels=["Nie", "Tak"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Nie", "Tak"],
        yticklabels=["Nie", "Tak"],
    )
    plt.title(f"Macierz pomyłek - {model_name}")
    plt.xlabel("Przewidziana etykieta")
    plt.ylabel("Prawdziwa etykieta")
    save_plot(file_name)


def create_feature_importance_plot(model_pipeline, numeric_features, categorical_features):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    encoded_categorical_features = (
        preprocessor
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )

    feature_names = list(numeric_features) + list(encoded_categorical_features)
    importances = model.feature_importances_

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    print("\nNajważniejsze cechy według modelu Random Forest:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance,
        x="importance",
        y="feature",
        color="#4C78A8",
        edgecolor="black",
    )
    plt.title("Najważniejsze cechy w modelu Random Forest")
    plt.xlabel("Ważność cechy")
    plt.ylabel("Cecha")
    save_plot("feature_importance_random_forest.png")


def run_machine_learning(df):
    X, y, preprocessor, numeric_features, categorical_features = prepare_ml_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Regresja logistyczna": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    print("\nWyniki modeli uczenia maszynowego")
    print("=" * 50)

    best_model_name = None
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{model_name}")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        file_name = (
            f"confusion_matrix_{model_name.lower().replace(' ', '_')}"
            .replace("ą", "a")
            .replace("ę", "e")
            .replace("ł", "l")
            .replace("ó", "o")
            .replace("ś", "s")
            .replace("ż", "z")
            .replace("ź", "z")
            .replace("ć", "c")
            .replace("ń", "n")
            + ".png"
        )

        create_confusion_matrix_plot(y_test, y_pred, model_name, file_name)

        if model_name == "Random Forest":
            create_feature_importance_plot(
                pipeline,
                numeric_features,
                categorical_features,
            )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = pipeline

    print("\nNajlepszy model:")
    print(f"{best_model_name} z accuracy: {best_accuracy:.3f}")

    return best_model_name, best_model


def main():
    setup_style()

    df = load_data(DATA_PATH)
    df = clean_data(df)

    print_dataset_overview(df)
    print_grouped_statistics(df)

    correlation_matrix = get_correlation_matrix(df)

    create_visualizations(df, correlation_matrix)

    run_machine_learning(df)


if __name__ == "__main__":
    main()