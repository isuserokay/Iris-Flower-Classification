
import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd


class IrisGUI:
    FEATURES = [
        ("Sepal Length (cm)", "sepal_length"),
        ("Sepal Width (cm)", "sepal_width"),
        ("Petal Length (cm)", "petal_length"),
        ("Petal Width (cm)", "petal_width"),
    ]

    SPECIES_NAMES = ["Setosa", "Versicolor", "Virginica"]

    MODEL_FILE = os.path.join(
        os.path.dirname(__file__), "iris_knn_model.pkl"
    )

    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classification")
        self.root.resizable(False, False)
        self.root.configure(bg="#F3F0FF")

        self.model = None
        self.entries = {}
        self.result = tk.StringVar(value="Result will appear here")

        self.create_widgets()
        self.load_model()

    def load_model(self):
        # Load the trained model
        try:
            self.model = joblib.load(self.MODEL_FILE)
        except Exception as error:
            messagebox.showerror(
                "Model error",
                "Could not load iris_knn_model.pkl.\n"
                "Make sure it is in the same folder as this file.\n\n"
                f"Details: {error}"
            )

    def create_widgets(self):
        # Create the window
        frame = tk.Frame(self.root, bg="#F3F0FF")
        frame.pack(padx=20, pady=20)

        title = tk.Label(
            frame,
            text="🪻 Iris Species Predictor 🪻",
            font=("Times", 18, "bold"),
            bg="#F3F0FF",
            fg="#513A91"
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 15))

        for i, (label_text, key) in enumerate(self.FEATURES):
            label = tk.Label(
                frame,
                text=label_text,
                bg="#F3F0FF",
                fg="#34304A"
            )
            label.grid(
                row=i + 1, column=0,
                sticky="w", padx=(0, 10), pady=5
            )

            entry = ttk.Entry(frame, width=15)
            entry.grid(row=i + 1, column=1, pady=5)

            self.entries[key] = entry

        # Add the buttons
        predict_button = ttk.Button(
            frame, text="Predict", command=self.on_predict
        )
        predict_button.grid(row=5, column=0, pady=15)

        clear_button = ttk.Button(
            frame, text="Clear", command=self.clear_fields
        )
        clear_button.grid(row=5, column=1, pady=15)

        # Show the prediction
        result_label = tk.Label(
            frame,
            textvariable=self.result,
            font=("Times", 18, "bold"),
            bg="#F3F0FF",
            fg="#513A91"
        )
        result_label.grid(row=6, column=0, columnspan=2)

    def get_values(self):
        # Check the input values
        values = []

        for label_text, key in self.FEATURES:
            text = self.entries[key].get().strip()

            try:
                value = float(text)
            except ValueError:
                messagebox.showwarning(
                    "Invalid input",
                    f"{label_text} must be a number!"
                )
                return None

            if value <= 0:
                messagebox.showwarning(
                    "Invalid input",
                    f"{label_text} must be greater than 0!"
                )
                return None

            values.append(value)

        return values

    def on_predict(self):
        # Check if the model is ready
        if self.model is None:
            messagebox.showerror(
                "Model error",
                "The model is not loaded, so I can't predict."
            )
            return

        # Get the measurements
        values = self.get_values()
        if values is None:
            return

        # Prepare the data
        columns = [key for label_text, key in self.FEATURES]
        row = pd.DataFrame([values], columns=columns)

        try:
            prediction = self.model.predict(row)
        except Exception as error:
            messagebox.showerror(
                "Prediction error",
                f"Could not predict:\n{error}"
            )
            return

        species = self.SPECIES_NAMES[prediction[0]]
        self.result.set(f"Predicted species: {species}")

    def clear_fields(self):
        # Clear the input fields
        for entry in self.entries.values():
            entry.delete(0, tk.END)

        self.result.set("Result will appear here!")


if __name__ == "__main__":
    window = tk.Tk()
    app = IrisGUI(window)
    window.mainloop()
