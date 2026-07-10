import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_sleep_health.csv"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

LIGHT_PURPLE = "#CDB4DB"
DARK_PURPLE = "#9D4EDD"
LIGHT_BLUE = "#BDE0FE"

DISORDER_PALETTE = {
    "No Disorder": "#BDE0FE",
    "Sleep Apnea": "#CDB4DB",
    "Insomnia": "#FFC8DD"
}


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:")
    print(df.shape)

    print("\nSleep disorder distribution:")
    print(df["Sleep_Disorder"].value_counts())

    print("\nNumeric summary:")
    print(df.describe())

    disorder_counts = df["Sleep_Disorder"].value_counts()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=disorder_counts.index,
        y=disorder_counts.values,
        hue=disorder_counts.index,
        palette=DISORDER_PALETTE,
        legend=False
    )
    plt.title("Rozkład zaburzeń snu")
    plt.xlabel("Zaburzenie snu")
    plt.ylabel("Liczba osób")
    save_plot("01_sleep_disorder_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Sleep_Disorder",
        y="Sleep_Duration",
        hue="Sleep_Disorder",
        palette=DISORDER_PALETTE,
        legend=False
    )
    plt.title("Czas snu a zaburzenia snu")
    plt.xlabel("Zaburzenie snu")
    plt.ylabel("Czas snu")
    save_plot("02_sleep_duration_by_disorder.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Sleep_Disorder",
        y="Stress_Level",
        hue="Sleep_Disorder",
        palette=DISORDER_PALETTE,
        legend=False
    )
    plt.title("Poziom stresu a zaburzenia snu")
    plt.xlabel("Zaburzenie snu")
    plt.ylabel("Poziom stresu")
    save_plot("03_stress_level_by_disorder.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Sleep_Disorder",
        y="Quality_of_Sleep",
        hue="Sleep_Disorder",
        palette=DISORDER_PALETTE,
        legend=False
    )
    plt.title("Jakość snu a zaburzenia snu")
    plt.xlabel("Zaburzenie snu")
    plt.ylabel("Jakość snu")
    save_plot("04_sleep_quality_by_disorder.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Sleep_Disorder",
        y="Daily_Steps",
        hue="Sleep_Disorder",
        palette=DISORDER_PALETTE,
        legend=False
    )
    plt.title("Liczba kroków dziennie a zaburzenia snu")
    plt.xlabel("Zaburzenie snu")
    plt.ylabel("Liczba kroków dziennie")
    save_plot("05_daily_steps_by_disorder.png")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        cmap="Purples",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Mapa korelacji zmiennych numerycznych")
    save_plot("06_correlation_heatmap.png")
