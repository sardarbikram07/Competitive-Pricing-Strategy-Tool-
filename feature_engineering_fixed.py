import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import traceback

# Suppress warnings but keep error messages
warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('data/feature_engineered', exist_ok=True)
os.makedirs('visualizations/feature_importance', exist_ok=True)

# Step 1: Load the original clean dataset
print("Loading and processing clean dataset...")
try:
    df = pd.read_csv('clean_dataset.csv')
    print(f"Dataset loaded with shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

# Define required columns for feature engineering
required_columns = ['discounted_price', 'actual_price', 'manufacturing_cost', 
                   'value_score', 'brand_strength', 'technology_level', 
                   'feature_count', 'price_elasticity', 'estimated_units_sold', 'rating', 'category']

# Check if required columns exist
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Error: Dataset is missing required columns: {missing_cols}")
    exit(1)

# Step 2: Identify and segment data by product categories
try:
    categories = df['category'].unique()
    print(f"Found {len(categories)} categories: {categories}")

    # Create a dictionary to store category dataframes
    category_dfs = {}

    # For each category, create a segment
    for category in categories:
        # Select rows for this category
        category_df = df[df['category'] == category].copy()
        
        # Check if category has enough samples
        if len(category_df) < 30:  # Minimum required for statistical significance
            print(f"Warning: Category {category} has only {len(category_df)} samples, which may be too few for reliable modeling")
        
        # Store in dictionary if we have data
        if len(category_df) > 0:
            category_dfs[category] = category_df
            print(f"Created segment for {category} with {len(category_df)} products")
except Exception as e:
    print(f"Error during category segmentation: {str(e)}")
    traceback.print_exc()
    exit(1)

# Step 3: Handle missing values in each category segment
print("\nHandling missing values for each category...")
for category, category_df in category_dfs.items():
    try:
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
                    
            # For categorical columns, fill with mode
            cat_cols = category_df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if col in cols_with_missing:
                    mode_val = category_df[col].mode()[0]
                    category_df[col].fillna(mode_val, inplace=True)
                    print(f"  Filled {col} with mode: {mode_val}")
        else:
            print(f"No missing values in {category}")
    except Exception as e:
        print(f"Error handling missing values for {category}: {str(e)}")
        continue

# Step 4: Handle negative or zero prices
print("\nChecking for negative or zero prices...")
for category, category_df in category_dfs.items():
    try:
        # Check for negative or zero prices
        neg_prices = category_df[category_df['discounted_price'] <= 0]
        if len(neg_prices) > 0:
            print(f"Found {len(neg_prices)} negative/zero prices in {category}")
            min_positive_price = category_df[category_df['discounted_price'] > 0]['discounted_price'].min()
            for idx in neg_prices.index:
                # Set to 10% markup over manufacturing cost or minimum positive price, whichever is greater
                mfg_cost = category_df.loc[idx, 'manufacturing_cost']
                new_price = max(mfg_cost * 1.1, min_positive_price)
                category_df.loc[idx, 'discounted_price'] = new_price
                print(f"  Fixed negative/zero price: set to {new_price:.2f} (was {neg_prices.loc[idx, 'discounted_price']:.2f})")
    except Exception as e:
        print(f"Error handling negative prices for {category}: {str(e)}")
        continue

# Step 5: Split data into training and testing sets for each category
print("\nSplitting data into training and testing sets...")
category_train_dfs = {}
category_test_dfs = {}

for category, category_df in category_dfs.items():
    try:
        # Verify minimum number of samples for splitting
        if len(category_df) < 50:
            print(f"Warning: {category} has only {len(category_df)} samples, which may be too few for reliable train/test splitting")
            if len(category_df) < 30:
                print(f"Error: {category} has too few samples ({len(category_df)}) for modeling. Skipping.")
                continue
                
        # Define features and target
        X = category_df.drop(['discounted_price'], axis=1)
        y = category_df['discounted_price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create complete dataframes with target
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Verify minimum samples after splitting
        if len(train_df) < 30 or len(test_df) < 10:
            print(f"Warning: After splitting, {category} has too few samples in train ({len(train_df)}) or test ({len(test_df)})")
        
        # Store in dictionaries
        category_train_dfs[category] = train_df
        category_test_dfs[category] = test_df
        
        print(f"Split for {category}:")
        print(f"  Training set: {len(train_df)} samples")
        print(f"  Testing set: {len(test_df)} samples")
    except Exception as e:
        print(f"Error during train/test splitting for {category}: {str(e)}")
        continue

# Step 6: Calculate category-specific pricing benchmarks
print("\nCalculating category-specific pricing benchmarks...")
category_benchmarks = {}

for category, train_df in category_train_dfs.items():
    try:
        # Calculate pricing benchmarks based on training data only
        benchmarks = {
            'median_price': train_df['discounted_price'].median(),
            'mean_price': train_df['discounted_price'].mean(),
            'min_price': train_df['discounted_price'].min(),
            'max_price': train_df['discounted_price'].max(),
            'q25_price': train_df['discounted_price'].quantile(0.25),
            'q75_price': train_df['discounted_price'].quantile(0.75),
            'lowest_quartile_upper_bound': train_df['discounted_price'].quantile(0.25),
            'median_manufacturing_cost': train_df['manufacturing_cost'].median(),
            'mean_profit_margin': (train_df['discounted_price'] / np.maximum(train_df['manufacturing_cost'], 1e-6)).mean(),
            'competitive_price_factor': train_df['discounted_price'].quantile(0.25) / np.maximum(train_df['manufacturing_cost'].median(), 1e-6)
        }
        
        # Verify benchmark values are valid
        invalid_benchmarks = [k for k, v in benchmarks.items() if pd.isna(v)]
        if invalid_benchmarks:
            print(f"Warning: {category} has invalid benchmark values for: {invalid_benchmarks}")
            for key in invalid_benchmarks:
                if key in ['q25_price', 'q75_price', 'lowest_quartile_upper_bound']:
                    # Use alternative percentiles or mean for quartiles
                    if key == 'q25_price' or key == 'lowest_quartile_upper_bound':
                        benchmarks[key] = train_df['discounted_price'].mean() * 0.75
                    else:
                        benchmarks[key] = train_df['discounted_price'].mean() * 1.25
                else:
                    # Use fallback values for other metrics
                    benchmarks[key] = train_df['discounted_price'].mean() if 'price' in key else 1.0
                print(f"  Fixed invalid {key} with value: {benchmarks[key]:.2f}")
        
        category_benchmarks[category] = benchmarks
        print(f"{category} benchmarks:")
        print(f"  Median price: {benchmarks['median_price']:.2f}")
        print(f"  Lowest quartile upper bound: {benchmarks['lowest_quartile_upper_bound']:.2f}")
        print(f"  Mean profit margin: {benchmarks['mean_profit_margin']:.2f}x")
        print(f"  Competitive price factor: {benchmarks['competitive_price_factor']:.4f}")
    except Exception as e:
        print(f"Error calculating benchmarks for {category}: {str(e)}")
        # Create default benchmarks as fallback
        median_price = train_df['discounted_price'].median() or train_df['discounted_price'].mean() or 1000
        category_benchmarks[category] = {
            'median_price': median_price,
            'mean_price': median_price,
            'min_price': median_price * 0.5,
            'max_price': median_price * 2.0,
            'q25_price': median_price * 0.75,
            'q75_price': median_price * 1.25,
            'lowest_quartile_upper_bound': median_price * 0.75,
            'median_manufacturing_cost': median_price * 0.5,
            'mean_profit_margin': 2.0,
            'competitive_price_factor': 0.75
        }
        print(f"Used fallback benchmarks for {category} due to calculation error")

# Step 7: Create new features for each category
print("\nCreating new features for each category...")

# Helper function for feature engineering
def engineer_features(df, category, benchmarks):
    try:
        # Create a copy to avoid modifying the original
        df_new = df.copy()
        
        # Create pricing strategy features focusing on customer attraction
        
        # 1. Price Competitiveness Ratio - how much below the median
        df_new['price_competitiveness_ratio'] = df_new['discounted_price'] / np.maximum(benchmarks['median_price'], 1e-6)
        
        # 2. Profit margin ratio (relative to manufacturing cost)
        df_new['profit_margin_ratio'] = df_new['discounted_price'] / np.maximum(df_new['manufacturing_cost'], 1e-6)
        
        # 3. Competitor undercut potential (how much room to undercut competitors and stay profitable)
        df_new['competitor_undercut_potential'] = (df_new['discounted_price'] - (df_new['manufacturing_cost'] * 1.1)) / np.maximum(df_new['discounted_price'], 1e-6)
        
        # 4. Price segmentation features
        df_new['is_budget_segment'] = (df_new['discounted_price'] <= benchmarks['q25_price']).astype(int)
        df_new['is_mid_segment'] = ((df_new['discounted_price'] > benchmarks['q25_price']) & 
                                  (df_new['discounted_price'] <= benchmarks['q75_price'])).astype(int)
        df_new['is_premium_segment'] = (df_new['discounted_price'] > benchmarks['q75_price']).astype(int)
        
        # 5. Customer attraction score - higher means more attractive to customers
        # Safely calculate using maximum values to avoid division by zero
        max_brand_strength = np.maximum(df_new['brand_strength'].max(), 1e-6)
        df_new['customer_attraction_score'] = (
            df_new['value_score'] + 
            (df_new['brand_strength'] / max_brand_strength) -  # normalize brand strength
            (df_new['price_competitiveness_ratio'] * 0.5)  # lower price increases score
        )
        
        # 6. Feature-to-price ratio (more features for the price)
        df_new['feature_to_price_ratio'] = df_new['feature_count'] / np.maximum(df_new['discounted_price'], 1e-6)
        
        # 7. Technology value proposition
        df_new['tech_value_proposition'] = df_new['technology_level'] / np.maximum(df_new['price_competitiveness_ratio'], 1e-6)
        
        # 8. Market positioning features
        df_new['relative_to_lowest_quartile'] = df_new['discounted_price'] / np.maximum(benchmarks['q25_price'], 1e-6)
        
        # 9. Value-adjusted price (price adjusted by the perceived value)
        df_new['value_adjusted_price'] = df_new['discounted_price'] * (1 - df_new['value_score'])
        
        # 10. Category-specific price index
        df_new['category_price_index'] = df_new['discounted_price'] / np.maximum(benchmarks['median_price'], 1e-6)
        
        # 11. Combined demand indicators
        df_new['demand_indicator'] = (df_new['estimated_units_sold'] * df_new['rating']) / np.maximum(df_new['discounted_price'], 1e-6)
        
        # 12. Brand strength to price ratio
        df_new['brand_power_price_ratio'] = df_new['brand_strength'] / np.maximum((df_new['price_competitiveness_ratio'] * 10), 1e-6)
        
        # 13. Calculate recommended pricing for new sellers (15-25% below average)
        # The model will learn how to adjust this further
        df_new['new_seller_competitive_price'] = df_new['discounted_price'] * 0.8  # 20% discount
        
        # 14. Sustainable price (minimum price to maintain 10% profit margin)
        df_new['sustainable_price'] = df_new['manufacturing_cost'] * 1.1
        
        # 15. Ideal customer attraction price (minimum of competitor pricing and sustainable price plus small margin)
        df_new['ideal_customer_attraction_price'] = np.maximum(
            benchmarks['lowest_quartile_upper_bound'] * 0.85,  # 15% below lowest quartile
            df_new['manufacturing_cost'] * 1.1  # minimum 10% margin
        )
        
        # 16. Market penetration potential
        discount_depth = 1 - (df_new['discounted_price'] / np.maximum(df_new['actual_price'], 1e-6))
        df_new['market_penetration_potential'] = discount_depth * df_new['value_score']
        
        # 17. Price elasticity adjusted value
        df_new['elasticity_adjusted_value'] = df_new['value_score'] / np.maximum(df_new['price_elasticity'], 1e-6)
        
        # Step to handle NaN values that might have been introduced
        numeric_cols = df_new.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df_new[col].isnull().sum() > 0:
                median_val = df_new[col].median()
                if pd.isna(median_val):  # If median is also NaN
                    median_val = 0
                df_new[col].fillna(median_val, inplace=True)
                print(f"  Fixed {df_new[col].isnull().sum()} NaN values in {col} with {median_val}")
        
        # Clip extreme values for all new features
        new_features = [col for col in df_new.columns if col not in df.columns]
        for col in new_features:
            if df_new[col].dtype in [np.float64, np.int64]:
                try:
                    q99 = df_new[col].quantile(0.99)
                    q01 = df_new[col].quantile(0.01)
                    
                    # Check if quantiles are valid
                    if pd.isna(q99) or pd.isna(q01):
                        print(f"  Warning: Cannot calculate reliable quantiles for {col}, using min/max instead")
                        q99 = df_new[col].max()
                        q01 = df_new[col].min()
                        
                        # If min/max also fail, use fallback values
                        if pd.isna(q99) or pd.isna(q01):
                            q99 = 1000
                            q01 = 0
                    
                    # Make sure q01 is less than q99
                    if q01 >= q99:
                        q01 = min(q01, 0)
                        q99 = max(q99, 1)
                        
                    df_new[col] = df_new[col].clip(q01, q99)  # Clipping at 1% and 99% quantiles
                except Exception as e:
                    print(f"  Warning: Error clipping values for {col}: {str(e)}")
        
        return df_new
    except Exception as e:
        print(f"Error in feature engineering for {category}: {str(e)}")
        traceback.print_exc()
        # Return original dataframe if an error occurs
        return df

# Apply feature engineering to each category
for category in categories:
    if category not in category_train_dfs or category not in category_test_dfs:
        print(f"Skipping feature engineering for {category} - missing train or test data")
        continue
        
    print(f"Engineering features for {category}...")
    
    try:
        # Get the benchmarks for this category
        benchmarks = category_benchmarks[category]
        
        # Engineer features for training and test sets
        train_df_engineered = engineer_features(category_train_dfs[category], category, benchmarks)
        test_df_engineered = engineer_features(category_test_dfs[category], category, benchmarks)
        
        # Verify feature engineering worked
        new_features_count = len(train_df_engineered.columns) - len(category_train_dfs[category].columns)
        if new_features_count == 0:
            print(f"Warning: No new features were created for {category}")
            continue
            
        # Check feature correlations with price - select only numeric columns
        try:
            numeric_cols = train_df_engineered.select_dtypes(include=['float64', 'int64']).columns
            if 'discounted_price' not in numeric_cols:
                print(f"Warning: discounted_price column not found in numeric columns for {category}")
                corr = pd.Series(dtype='float64')
            else:
                corr = train_df_engineered[numeric_cols].corr()['discounted_price'].sort_values(ascending=False)
            print(f"Top 5 features correlated with price:")
            print(corr.head(min(6, len(corr))).to_string())  # 6 includes the target itself, but handle case with fewer columns
            
            # Visualize feature correlations - only numeric columns
            plt.figure(figsize=(12, 10))
            corr_matrix = train_df_engineered[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title(f'Feature Correlations for {category}')
            plt.tight_layout()
            plt.savefig(f'visualizations/feature_importance/{category}_correlation_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Error creating correlation analysis for {category}: {str(e)}")
        
        # Save the engineered datasets
        train_df_engineered.to_csv(f'data/feature_engineered/{category}_train_features.csv', index=False)
        test_df_engineered.to_csv(f'data/feature_engineered/{category}_test_features.csv', index=False)
        
        print(f"  Added {new_features_count} new features")
        print(f"  Saved engineered datasets with {len(train_df_engineered.columns)} total features")
    except Exception as e:
        print(f"Error processing category {category}: {str(e)}")
        traceback.print_exc()

print("\nFeature engineering completed. Files saved to data/feature_engineered/ directory.")
print("Feature correlation visualizations saved to visualizations/feature_importance/ directory.")

# Step 8: Create a summary file with category benchmarks
try:
    benchmark_df = pd.DataFrame.from_dict(category_benchmarks, orient='index')
    benchmark_df.to_csv('data/feature_engineered/category_benchmarks.csv')
    print("Category benchmarks saved to data/feature_engineered/category_benchmarks.csv")
except Exception as e:
    print(f"Error saving category benchmarks: {str(e)}")

# Print execution summary
processed_categories = len([c for c in categories if c in category_train_dfs and c in category_test_dfs])
print(f"\nSummary: Processed {processed_categories} out of {len(categories)} categories successfully") 