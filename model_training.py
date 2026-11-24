from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

from data_loader import DataLoader


class ModelTraining:

    def __init__(self, data):
        self.data = data

        self.feature_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=3))
        ])

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self):
        """Split the dataset into training and testing sets."""

        X = self.data[self.feature_columns]
        y = self.data["target"]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

        print("\n========== DATA SPLITTING ==========")
        print("Training samples:", len(self.X_train))
        print("Testing samples:", len(self.X_test))

    def train_model(self):
        """Train the KNN classification model."""

        if self.X_train is None:
            raise ValueError("Split the data before training.")

        self.model.fit(self.X_train, self.y_train)

        print("\nKNN model trained successfully.")

    def evaluate_model(self):
        """Test the model and calculate accuracy."""

        if self.X_test is None:
            raise ValueError("Split the data before testing.")

        predictions = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, predictions)

        print("\n========== MODEL EVALUATION ==========")
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def save_model(self, filename="iris_knn_model.pkl"):
        """Save the trained model."""

        joblib.dump(self.model, filename)

        print(f"\nModel saved successfully as {filename}")

    def run_training(self):
        """Run the complete training and evaluation process."""

        self.split_data()
        self.train_model()
        accuracy = self.evaluate_model()
        self.save_model()

        return accuracy


if __name__ == "__main__":

    loader = DataLoader()
    data = loader.load_data()
    data = loader.validate_data()

    trainer = ModelTraining(data)
    trainer.run_training()