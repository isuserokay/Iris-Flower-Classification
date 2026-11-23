import pandas as pd
from sklearn.datasets import load_iris

class DataLoader:
    def __init__(self):
       self.data = None

    def load_data(self):
        """Load the Iris dataset and convert it into a DataFrame."""
        try:
            iris = load_iris()

            self.data = pd.DataFrame(
                iris.data,
                columns=[
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width"
                ]
            )

            # Add the target class and species name
            self.data["target"] = iris.target
            self.data["species"] = [
                iris.target_names[i] for i in iris.target
            ]

            print("Iris dataset loaded successfully.")
            return self.data

        except Exception as error:
            raise RuntimeError(
                f"Error while loading the dataset: {error}"
            )

    def validate_data(self):
        """Check the dataset structure and values."""
        if self.data is None:
            raise ValueError("Dataset has not been loaded.")

        required_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "target",
            "species"
        ]

        # Check required columns
        missing_columns = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        # Check data types and convert measurement columns
        feature_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        for col in feature_columns:
            self.data[col] = pd.to_numeric(
                self.data[col],errors="coerce"
            )

        # Replace invalid or non-positive measurements with NaN
        for col in feature_columns:
            self.data.loc[self.data[col] <= 0, col] = float("nan")

        # Remove rows without a species label
        self.data.dropna(subset=["target", "species"], inplace=True)

        # Fill missing numerical values using the median
        for col in feature_columns:
            self.data[col] = self.data[col].fillna(
                self.data[col].median()
            )

        # Final validation
        if self.data[feature_columns].isnull().any().any():
            raise ValueError("Missing values remain in the dataset.")

        if not (self.data[feature_columns] > 0).all().all():
            raise ValueError("Invalid measurement values found.")

        print("Data validation completed successfully.")
        return self.data

    def get_data(self):
        """Return the prepared dataset."""
        if self.data is None:
            raise ValueError("Load the dataset first.")

        return self.data



