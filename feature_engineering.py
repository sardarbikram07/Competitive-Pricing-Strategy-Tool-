import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory for processed features
os.makedirs('data/feature_engineered', exist_ok=True)
os.makedirs('visualizations/feature_importance', exist_ok=True)

# Step 1: Load the processed datasets for each category
print("Loading processed datasets...")
category_train_dfs = {}
category_test_dfs = {}

# Get list of all training files
train_files = glob.glob('data/processed/*_train.csv')
category_names = [os.path.basename(f).replace('_train.csv', '') for f in train_files]

# Load each category's training and test data
for category in category_names:
    train_path = f'data/processed/{category}_train.csv'
    test_path = f'data/processed/{category}_test.csv'
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        category_train_dfs[category] = pd.read_csv(train_path)
        category_test_dfs[category] = pd.read_csv(test_path)
        print(f"Loaded {category}: {len(category_train_dfs[category])} training samples, {len(category_test_dfs[category])} test samples")

# Step 2: Fix the negative/zero price in Home Office category
if 'Home Office' in category_train_dfs:
    # Check for negative prices in training set
    neg_prices_train = category_train_dfs['Home Office'][category_train_dfs['Home Office']['discounted_price'] <= 0]
    if len(neg_prices_train) > 0:
        print(f"Found {len(neg_prices_train)} negative/zero prices in Home Office training set")
        # Set minimum price based on manufacturing cost
        min_positive_price = category_train_dfs['Home Office'][category_train_dfs['Home Office']['discounted_price'] > 0]['discounted_price'].min()
        for idx in neg_prices_train.index:
            # Set to 10% markup over manufacturing cost or minimum positive price, whichever is greater
            mfg_cost = category_train_dfs['Home Office'].loc[idx, 'manufacturing_cost']
            new_price = max(mfg_cost * 1.1, min_positive_price)
            category_train_dfs['Home Office'].loc[idx, 'discounted_price'] = new_price
            print(f"  Fixed negative price: set to {new_price:.2f} (was {neg_prices_train.loc[idx, 'discounted_price']:.2f})")
    
    # Check for negative prices in test set
    neg_prices_test = category_test_dfs['Home Office'][category_test_dfs['Home Office']['discounted_price'] <= 0]
    if len(neg_prices_test) > 0:
        print(f"Found {len(neg_prices_test)} negative/zero prices in Home Office test set")
        min_positive_price = category_test_dfs['Home Office'][category_test_dfs['Home Office']['discounted_price'] > 0]['discounted_price'].min()
        for idx in neg_prices_test.index:
            mfg_cost = category_test_dfs['Home Office'].loc[idx, 'manufacturing_cost']
            new_price = max(mfg_cost * 1.1, min_positive_price)
            category_test_dfs['Home Office'].loc[idx, 'discounted_price'] = new_price
            print(f"  Fixed negative price: set to {new_price:.2f} (was {neg_prices_test.loc[idx, 'discounted_price']:.2f})")

# Step 3: Calculate category-specific pricing benchmarks
print("\nCalculating category-specific pricing benchmarks...")
category_benchmarks = {}

for category, train_df in category_train_dfs.items():
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
        'mean_profit_margin': (train_df['discounted_price'] / train_df['manufacturing_cost']).mean(),
        'competitive_price_factor': train_df['discounted_price'].quantile(0.25) / train_df['manufacturing_cost'].median()
    }
    
    category_benchmarks[category] = benchmarks
    print(f"{category} benchmarks:")
    print(f"  Median price: {benchmarks['median_price']:.6f}")
    print(f"  Lowest quartile upper bound: {benchmarks['lowest_quartile_upper_bound']:.6f}")
    print(f"  Mean profit margin: {benchmarks['mean_profit_margin']:.2f}x")
    print(f"  Competitive price factor: {benchmarks['competitive_price_factor']:.4f}")

# Step 4: Create new features for each category
print("\nCreating new features for each category...")

# Helper function for feature engineering
def engineer_features(df, category, benchmarks):
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Create pricing strategy features focusing on customer attraction
    
    # 1. Price Competitiveness Ratio - how much below the median
    df_new['price_competitiveness_ratio'] = df_new['discounted_price'] / benchmarks['median_price']
    
    # 2. Profit margin ratio (relative to manufacturing cost)
    df_new['profit_margin_ratio'] = df_new['discounted_price'] / df_new['manufacturing_cost']
    
    # 3. Competitor undercut potential (how much room to undercut competitors and stay profitable)
    df_new['competitor_undercut_potential'] = (df_new['discounted_price'] - (df_new['manufacturing_cost'] * 1.1)) / df_new['discounted_price']
    
    # 4. Price segmentation features
    df_new['is_budget_segment'] = (df_new['discounted_price'] <= benchmarks['q25_price']).astype(int)
    df_new['is_mid_segment'] = ((df_new['discounted_price'] > benchmarks['q25_price']) & 
                              (df_new['discounted_price'] <= benchmarks['q75_price'])).astype(int)
    df_new['is_premium_segment'] = (df_new['discounted_price'] > benchmarks['q75_price']).astype(int)
    
    # 5. Customer attraction score - higher means more attractive to customers
    # (based on value score, brand strength, price competitiveness)
    df_new['customer_attraction_score'] = (
        df_new['value_score'] + 
        (df_new['brand_strength'] / 50) -  # normalize brand strength contribution
        (df_new['price_competitiveness_ratio'] * 0.5)  # lower price increases score
    )
    
    # 6. Feature-to-price ratio (more features for the price)
    df_new['feature_to_price_ratio'] = df_new['feature_count'] / df_new['discounted_price']
    
    # 7. Technology value proposition
    df_new['tech_value_proposition'] = df_new['technology_level'] / df_new['price_competitiveness_ratio']
    
    # 8. Market positioning features
    df_new['relative_to_lowest_quartile'] = df_new['discounted_price'] / benchmarks['q25_price']
    
    # 9. Value-adjusted price (price adjusted by the perceived value)
    df_new['value_adjusted_price'] = df_new['discounted_price'] * (1 - df_new['value_score'])
    
    # 10. Category-specific price index
    df_new['category_price_index'] = df_new['discounted_price'] / benchmarks['median_price']
    
    # 11. Combined demand indicators
    df_new['demand_indicator'] = (df_new['estimated_units_sold'] * df_new['rating']) / df_new['discounted_price']
    
    # 12. Brand strength to price ratio
    df_new['brand_power_price_ratio'] = df_new['brand_strength'] / (df_new['price_competitiveness_ratio'] * 10)
    
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
    
    return df_new

# Apply feature engineering to each category
for category in category_names:
    print(f"Engineering features for {category}...")
    
    # Get the benchmarks for this category
    benchmarks = category_benchmarks[category]
    
    # Engineer features for training and test sets
    train_df_engineered = engineer_features(category_train_dfs[category], category, benchmarks)
    test_df_engineered = engineer_features(category_test_dfs[category], category, benchmarks)
    
    # Check feature correlations with price
    corr = train_df_engineered.corr()['discounted_price'].sort_values(ascending=False)
    print(f"Top 5 features correlated with price:")
    print(corr.head(6).to_string())  # 6 includes the target itself
    
    # Visualize feature correlations
    plt.figure(figsize=(12, 10))
    corr_matrix = train_df_engineered.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlations for {category}')
    plt.tight_layout()
    plt.savefig(f'visualizations/feature_importance/{category}_correlation_matrix.png')
    plt.close()
    
    # Save the engineered datasets
    train_df_engineered.to_csv(f'data/feature_engineered/{category}_train_features.csv', index=False)
    test_df_engineered.to_csv(f'data/feature_engineered/{category}_test_features.csv', index=False)
    
    print(f"  Added {len(train_df_engineered.columns) - len(category_train_dfs[category].columns)} new features")
    print(f"  Saved engineered datasets with {len(train_df_engineered.columns)} total features")

print("\nFeature engineering completed. Files saved to data/feature_engineered/ directory.")
print("Feature correlation visualizations saved to visualizations/feature_importance/ directory.")

# Step 5: Create a summary file with category benchmarks
benchmark_df = pd.DataFrame.from_dict(category_benchmarks, orient='index')
benchmark_df.to_csv('data/feature_engineered/category_benchmarks.csv')
print("Category benchmarks saved to data/feature_engineered/category_benchmarks.csv") 