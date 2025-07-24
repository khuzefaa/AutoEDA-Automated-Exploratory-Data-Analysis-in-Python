import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Set file path
file_path = os.path.join("data", "UCI_Credit_Card.csv")

#Load dataset
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("File not found. Please check the path.")
print(df.head())
print(f"\n Shape: {df.shape}")

#Data Info
print("\n Data Types:\n", df.dtypes)
print("\n Missing Values:\n", df.isnull().sum())

#Clean Data (drop NA for now)
df_cleaned = df.dropna()

#Correlation Matrix
print("\n Correlation Matrix:")
correlation = df_cleaned.corr(numeric_only=True)
print(correlation)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#Outlier Detection (IQR)
print("\n Outlier Detection (IQR method):")
numeric_df = df_cleaned.select_dtypes(include=[np.number])
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

#Safer outlier comparison loop
outliers = pd.DataFrame(False, index=df_cleaned.index, columns=df_cleaned.columns)
for col in numeric_df.columns:
    q1 = Q1[col]
    q3 = Q3[col]
    iqr = IQR[col]
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers[col] = (df_cleaned[col] < lower) | (df_cleaned[col] > upper)

#Outlier Summary
print("\n Outliers Summary (True = outlier):")
print(outliers.sum())

#Visualizations

## Histogram for numeric columns
for col in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f'📊 Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

## Boxplot for numeric columns
for col in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_cleaned[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

## Pairplot
sns.pairplot(numeric_df)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.tight_layout()
plt.show()
