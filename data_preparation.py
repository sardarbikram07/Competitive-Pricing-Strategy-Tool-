import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Create directories for outputs
os.makedirs('data/processed', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Step 1: Load normalized dataset
print("Loading dataset...")
df = pd.read_csv('normalized_dataset.csv')
print(f"Dataset loaded with shape: {df.shape}")

# Print column names to understand the data structure
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Step 2: Identify and segment data by product categories
# First check what category columns we have (they should be one-hot encoded)
category_cols = [col for col in df.columns if col.startswith('category_')]
print(f"\nFound {len(category_cols)} category columns: {category_cols}")

# Create a dictionary to store category dataframes
category_dfs = {}

# For each category, create a segment
for category in category_cols:
    # Extract the category name from the column
    category_name = category.replace('category_', '')
    
    # Select rows where this category is 1
    category_df = df[df[category] == 1].copy()
    
    # Store in dictionary if we have data
    if len(category_df) > 0:
        category_dfs[category_name] = category_df
        print(f"Created segment for {category_name} with {len(category_df)} products")

# Step 3: Handle missing values in each category segment
print("\nHandling missing values for each category...")
for category, category_df in category_dfs.items():
    # Check missing values
    missing_count = category_df.isnull().sum()
    missing_percent = (missing_count / len(category_df)) * 100
    
    # Report columns with missing values
    cols_with_missing = missing_count[missing_count > 0]
    if len(cols_with_missing) > 0:
        print(f"\nMissing values in {category}:")
        for col, count in cols_with_missing.items():
            print(f"  {col}: {count} values ({missing_percent[col]:.2f}%)")
        
        # For numerical columns, fill with median
        num_cols = category_df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if col in cols_with_missing:
                median_val = category_df[col].median()
                category_df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} with median: {median_val:.2f}")
    else:
        print(f"No missing values in {category}")

# Step 4: Analyze data distribution for each category
print("\nAnalyzing data distributions...")
for category, category_df in category_dfs.items():
    print(f"\nSummary statistics for {category}:")
    
    # Basic statistics for numerical columns
    num_cols = category_df.select_dtypes(include=['float64', 'int64']).columns
    print(category_df[num_cols].describe().to_string())
    
    # Create distribution plots for discounted_price
    plt.figure(figsize=(10, 6))
    sns.histplot(data=category_df, x='discounted_price', kde=True)
    plt.title(f'Price Distribution for {category}')
    plt.savefig(f'visualizations/{category}_price_distribution.png')
    plt.close()

# Step 5: Check for outliers in pricing data
print("\nChecking for outliers in pricing data...")
for category, category_df in category_dfs.items():
    # Calculate IQR for price
    Q1 = category_df['discounted_price'].quantile(0.25)
    Q3 = category_df['discounted_price'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = category_df[(category_df['discounted_price'] < lower_bound) | 
                          (category_df['discounted_price'] > upper_bound)]
    
    print(f"\nOutliers in {category}:")
    print(f"  IQR range: {lower_bound:.2f} to {upper_bound:.2f}")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(category_df)*100:.2f}%)")
    
    # Create boxplot for visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=category_df['discounted_price'])
    plt.title(f'Price Boxplot for {category}')
    plt.savefig(f'visualizations/{category}_price_boxplot.png')
    plt.close()
    
    # For this analysis, we'll keep the outliers but flag them
    category_df['is_price_outlier'] = ((category_df['discounted_price'] < lower_bound) | 
                                      (category_df['discounted_price'] > upper_bound))

# Step 6: Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
for category, category_df in category_dfs.items():
    # Define features and target
    X = category_df.drop(['discounted_price'], axis=1)
    y = category_df['discounted_price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Split for {category}:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    # Save the splits
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(f'data/processed/{category}_train.csv', index=False)
    test_df.to_csv(f'data/processed/{category}_test.csv', index=False)

# Step 7: Verify data quality
print("\nVerifying data quality...")
for category, category_df in category_dfs.items():
    # Check for remaining missing values
    missing = category_df.isnull().sum().sum()
    if missing > 0:
        print(f"Warning: {category} still has {missing} missing values!")
    else:
        print(f"{category} has no missing values.")
    
    # Check for negative prices
    neg_prices = (category_df['discounted_price'] <= 0).sum()
    if neg_prices > 0:
        print(f"Warning: {category} has {neg_prices} products with zero or negative prices!")
    else:
        print(f"{category} has no negative prices.")
    
    # Check feature correlations with price
    corr = category_df.corr()['discounted_price'].sort_values(ascending=False)
    print(f"\nTop 5 correlations with price for {category}:")
    print(corr.head(6).to_string())  # 6 because one is the target itself

print("\nData preparation complete! Files saved to data/processed/ directory.")
print("Visualizations saved to visualizations/ directory.") 