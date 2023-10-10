import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=';')
print(wine_data.head())
print(wine_data.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=wine_data)
plt.title('Distribution of Wine Quality Scores')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()
correlation_matrix = wine_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
quality_counts = wine_data['quality'].value_counts()
print("Distribution of Wine Quality Scores:")
print(quality_counts)
correlations = wine_data.corr()['quality'].drop('quality')
print("\nCorrelation of Features with Wine Quality:")
print(correlations)
important_features = correlations[abs(correlations) > 0.2]
print("\nImportant Factors Influencing Wine Quality:")
print(important_features)
