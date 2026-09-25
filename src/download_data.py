from pathlib import Path

from sklearn.datasets import fetch_california_housing


def download_dataset():
    # Load the California Housing dataset as a pandas DataFrame
    housing = fetch_california_housing(as_frame=True)

    # Features
    df = housing.data.copy()

    # Add target column
    df["MedHouseVal"] = housing.target

    # Create the output path
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "california_housing.csv"

    # Save dataset
    df.to_csv(output_file, index=False)

    print("Dataset downloaded successfully.")
    print(f"Saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    download_dataset()