# Analyzing-the-correlation-between-students-nationality-and-grades

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder


# Step 1: Create a mock dataset
def create_dataset():
    """
    Creates a mock dataset representing students at IE University.

    Returns:
        DataFrame: A pandas DataFrame with columns for StudentID, Height (in cm), Nationality, and Grade.
    """
    np.random.seed(0)  # Ensuring reproducibility
    student_ids = np.arange(1, 101)  # Generating 100 student IDs
    heights = np.random.normal(170, 10, 100).astype(int)  # Normal distribution around 170cm with SD of 10
    nationalities = np.random.choice(['Spain', 'Italy', 'Germany', 'France', 'Portugal'], 100)  # Random nationalities
    grades = np.random.normal(75, 10, 100).astype(int)  # Normal distribution around 75/100 with SD of 10

    return pd.DataFrame({'StudentID': student_ids, 'Height': heights, 'Nationality': nationalities, 'Grade': grades})


# Step 2: Analyze Average Height
def analyze_average_height(df):
    """
    Calculates the average height of students.

    Args:
        df (DataFrame): The students' DataFrame.

    Returns:
        float: The average height of students.
    """
    return df['Height'].mean()


# Step 4: Analyze correlation between nationality and grades
def analyze_correlation(df):
    """
    Analyzes the correlation between students' nationalities and their grades.

    Args:
        df (DataFrame): The students' DataFrame.

    Returns:
        DataFrame: A DataFrame with nationality, encoded features, and the correlation value.
    """
    # Encoding nationalities using OneHotEncoder for a more appropriate categorical treatment
    encoder = OneHotEncoder(sparse=False)
    nationalities_encoded = encoder.fit_transform(df[['Nationality']])

    # Calculating correlation for each nationality category with grades
    correlations = []
    for i, category in enumerate(encoder.categories_[0]):
        correlation, _ = pearsonr(nationalities_encoded[:, i], df['Grade'])
        correlations.append((category, correlation))

    return pd.DataFrame(correlations, columns=['Nationality', 'Correlation'])


# Main function to run the analyses
def main():
    df = create_dataset()
    print(f"Average Height: {analyze_average_height(df):.2f} cm")

    correlation_df = analyze_correlation(df)
    print("\nCorrelation between nationality and grades:")
    print(correlation_df)

# The main function is commented out to prevent automatic execution in this context
# When running the script, uncomment the line below
# main()
