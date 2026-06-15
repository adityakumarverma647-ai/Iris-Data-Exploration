import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

df.hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.savefig("histogram.png")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df.iloc[:, :-1])
plt.title("Boxplot of Features")
plt.savefig("boxplot.png")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

species_names = {0:"Setosa",1:"Versicolor",2:"Virginica"}
df["species_name"] = df["species"].map(species_names)

sns.pairplot(df, hue="species_name")
plt.savefig("pairplot.png")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="species_name", data=df)
plt.title("Species Distribution")
plt.savefig("species_distribution.png")
plt.show()

print("Project Completed Successfully")
