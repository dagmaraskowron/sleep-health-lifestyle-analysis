import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "sleep_health_lifestyle.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean_sleep_health.csv"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


def split_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    blood_pressure = df["Blood_Pressure"].str.split("/", expand=True)

    df["Systolic_BP"] = blood_pressure[0].astype(int)
    df["Diastolic_BP"] = blood_pressure[1].astype(int)

    df = df.drop(columns=["Blood_Pressure"])

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = clean_column_names(df)

    df = df.drop_duplicates()

    df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("No Disorder")

    df = split_blood_pressure(df)

    df = df.drop(columns=["Person_ID"])

    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)

    print("Initial shape:")
    print(df.shape)

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    print("\nDuplicated rows:")
    print(df.duplicated().sum())

    clean_df = clean_data(df)

    print("\nClean shape:")
    print(clean_df.shape)

    print("\nColumns after cleaning:")
    print(clean_df.columns.tolist())

    print("\nMissing values after cleaning:")
    print(clean_df.isnull().sum())

    print("\nSleep disorder distribution:")
    print(clean_df["Sleep_Disorder"].value_counts())

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)