import matplotlib.pyplot as plt
from data_loader import DataLoader

class DataExploration:
   def __init__(self, data):
      self.data = data

   def show_basic_information(self):
       """Display basic information about the dataset."""
       print("\n========== DATASET OVERVIEW ==========")

       print("\nFirst 5 rows:")
       print(self.data.head())

       print("\nDataset shape:")
       print(self.data.shape)

       print("\nDataset information:")
       self.data.info()

       print("\nStatistical summary:")
       print(self.data.describe())

   def check_missing_values(self):
       """Display missing values in each column."""
       print("\n========== MISSING VALUES ==========")
       print(self.data.isnull().sum())

   def show_species_distribution(self):
       """Count the samples in each species."""
       print("\n========== SPECIES DISTRIBUTION ==========")
       print(self.data["species"].value_counts())

   def show_average_measurements(self):
       """Calculate average measurements for each species."""
       print("\n========== AVERAGE MEASUREMENTS ==========")

       averages = self.data.groupby("species")[
           [
               "sepal_length",
               "sepal_width",
               "petal_length",
               "petal_width"
           ]
       ].mean()

       print(averages)

   def plot_species_distribution(self):
       """Create a bar chart for species distribution."""
       counts = self.data["species"].value_counts()

       counts.plot(kind="bar", figsize=(7, 5))

       plt.title("Iris Species Distribution")
       plt.xlabel("Species")
       plt.ylabel("Number of Samples")
       plt.xticks(rotation=0)
       plt.tight_layout()
       plt.show()

   def plot_scatter(self):
       """Create a scatter plot of petal measurements."""
       species_list = self.data["species"].unique()

       for species in species_list:
           subset = self.data[
               self.data["species"] == species
               ]

           plt.scatter(
               subset["petal_length"],
               subset["petal_width"],
               label=species
           )

       plt.title("Petal Length vs Petal Width")
       plt.xlabel("Petal Length (cm)")
       plt.ylabel("Petal Width (cm)")
       plt.legend()
       plt.tight_layout()
       plt.show()

   def plot_histograms(self):
       """Create histograms for all four measurements."""
       columns = [
           "sepal_length",
           "sepal_width",
           "petal_length",
           "petal_width"
       ]

       self.data[columns].hist(
           figsize=(10, 8),
           bins=15,
           edgecolor="black"
       )

       plt.suptitle("Distribution of Iris Measurements")
       plt.tight_layout()
       plt.show()

   def plot_boxplots(self):
       """Compare measurement distributions across species."""
       columns = [
           "sepal_length",
           "sepal_width",
           "petal_length",
           "petal_width"
       ]

       for column in columns:
           self.data.boxplot(
               column=column,
               by="species",
               figsize=(7, 5)
           )

           plt.title(f"{column} by Species")
           plt.suptitle("")
           plt.xlabel("Species")
           plt.ylabel("Measurement (cm)")
           plt.tight_layout()
           plt.show()

   def run_exploration(self):
       """Run all data exploration tasks."""
       self.show_basic_information()
       self.check_missing_values()
       self.show_species_distribution()
       self.show_average_measurements()

       self.plot_species_distribution()
       self.plot_scatter()
       self.plot_histograms()
       self.plot_boxplots()


if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_data()
    data = loader.validate_data()
    exploration = DataExploration(data)
    exploration.run_exploration()