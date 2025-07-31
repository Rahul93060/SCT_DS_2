# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/Rahul/Documents/New folder (3)/train.csv")  # Update path if needed

# ----- Data Cleaning -----

# Drop 'Cabin' due to too many missing values
df_cleaned = df.drop(columns=['Cabin'])

# Fill missing 'Age' values with median
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with mode
df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)

# Check if any missing values remain
print("Missing values after cleaning:\n", df_cleaned.isnull().sum())

# ----- Summary Statistics -----
print("\nSummary statistics:\n", df_cleaned.describe(include='all'))

# ----- EDA: Exploratory Data Analysis -----

# Set seaborn style
sns.set(style="whitegrid")

# 1. Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='Survived', palette='Set2')
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='Sex', hue='Survived', palette='pastel')
plt.title('Survival by Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='Pclass', hue='Survived', palette='Set3')
plt.title('Survival by Passenger Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df_cleaned, x='Age', hue='Survived', fill=True)
plt.title('Age Distribution by Survival')
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
