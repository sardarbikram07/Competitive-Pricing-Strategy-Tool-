import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Suppress warnings but keep error messages
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('visualizations/distributions', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def load_dataset(file_path='clean_dataset.csv'):
    """
    Load the dataset and perform basic validation
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_columns = ['discounted_price', 'actual_price', 'discount_percentage', 
                           'rating', 'rating_count', 'category', 'manufacturing_cost']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def clean_dataset(df):
    """
    Clean the dataset by handling missing values, duplicates, and basic data issues
    """
    logger.info("Starting data cleaning process")
    initial_rows = df.shape[0]
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    missing_values = df_clean.isnull().sum()
    logger.info(f"Missing values before cleaning:\n{missing_values[missing_values > 0]}")
    
    # For numerical columns, fill with median
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.info(f"Filled {df_clean[col].isnull().sum()} missing values in {col} with median: {median_val}")
    
    # For categorical columns, fill with mode
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            logger.info(f"Filled {df_clean[col].isnull().sum()} missing values in {col} with mode: {mode_val}")
    
    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - df_clean.shape[0]
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Validate price data and handle negative prices
    price_columns = ['discounted_price', 'actual_price', 'manufacturing_cost']
    for col in price_columns:
        if col in df_clean.columns:
            neg_prices = (df_clean[col] < 0).sum()
            if neg_prices > 0:
                logger.warning(f"Found {neg_prices} negative values in {col}")
                # Replace negative prices with absolute values
                df_clean.loc[df_clean[col] < 0, col] = df_clean.loc[df_clean[col] < 0, col].abs()
                logger.info(f"Converted {neg_prices} negative prices to positive in {col}")
    
    # Ensure discount percentage is between 0 and 100
    if 'discount_percentage' in df_clean.columns:
        invalid_discount = ((df_clean['discount_percentage'] < 0) | (df_clean['discount_percentage'] > 100)).sum()
        if invalid_discount > 0:
            logger.warning(f"Found {invalid_discount} invalid discount percentages")
            # Clip discount percentage to valid range
            df_clean['discount_percentage'] = df_clean['discount_percentage'].clip(0, 100)
    
    # Handle inconsistent categories by standardizing format
    if 'category' in df_clean.columns:
        df_clean['category'] = df_clean['category'].str.strip().str.title()
        logger.info(f"Standardized {len(df_clean['category'].unique())} category names")
    
    # Validate that actual price >= discounted price
    price_inconsistency = (df_clean['actual_price'] < df_clean['discounted_price']).sum()
    if price_inconsistency > 0:
        logger.warning(f"Found {price_inconsistency} cases where actual price < discounted price")
        # Fix by swapping values
        idx = df_clean['actual_price'] < df_clean['discounted_price']
        temp = df_clean.loc[idx, 'actual_price'].copy()
        df_clean.loc[idx, 'actual_price'] = df_clean.loc[idx, 'discounted_price']
        df_clean.loc[idx, 'discounted_price'] = temp
        logger.info(f"Fixed {price_inconsistency} inconsistent price pairs by swapping values")
    
    logger.info(f"Data cleaning complete. Rows before: {initial_rows}, Rows after: {df_clean.shape[0]}")
    return df_clean

def handle_outliers(df, price_cols=['discounted_price', 'actual_price', 'manufacturing_cost']):
    """
    Handle outliers in price columns using IQR-based capping instead of removal
    """
    logger.info("Starting outlier handling process")
    df_processed = df.copy()
    
    outlier_stats = {}
    
    # Handle outliers separately for each category
    categories = df_processed['category'].unique()
    
    for category in categories:
        category_data = df_processed[df_processed['category'] == category]
        logger.info(f"Processing outliers for category: {category} ({len(category_data)} products)")
        
        category_stats = {}
        
        for col in price_cols:
            # Calculate IQR for the column
            Q1 = category_data[col].quantile(0.25)
            Q3 = category_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers (using 1.5 IQR)
            lower_bound = max(0, Q1 - 1.5 * IQR)  # Never go below 0 for prices
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before capping
            outliers_lower = (category_data[col] < lower_bound).sum()
            outliers_upper = (category_data[col] > upper_bound).sum()
            total_outliers = outliers_lower + outliers_upper
            
            # Store statistics
            category_stats[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_lower': outliers_lower,
                'outliers_upper': outliers_upper,
                'total_outliers': total_outliers,
                'percentage_outliers': (total_outliers / len(category_data)) * 100
            }
            
            # Log outlier information
            logger.info(f"{category} - {col}: Found {total_outliers} outliers " +
                       f"({outliers_lower} below {lower_bound:.2f}, {outliers_upper} above {upper_bound:.2f})")
            
            # Cap outliers instead of removing them
            mask = (df_processed['category'] == category)
            df_processed.loc[mask & (df_processed[col] < lower_bound), col] = lower_bound
            df_processed.loc[mask & (df_processed[col] > upper_bound), col] = upper_bound
            
            logger.info(f"Capped outliers in {category} - {col}")
        
        outlier_stats[category] = category_stats
    
    # Save outlier statistics for future reference
    with open('logs/outlier_stats.json', 'w') as f:
        json.dump(outlier_stats, f, indent=4, default=str)
    
    logger.info("Outlier handling complete")
    return df_processed, outlier_stats

def create_price_features(df):
    """
    Create advanced price-related features including transformations and relative metrics
    """
    logger.info("Creating price-related features")
    df = df.copy()
    
    # Basic price-related ratios
    df['price_to_cost_ratio'] = df['discounted_price'] / df['manufacturing_cost']
    df['margin_percentage'] = ((df['discounted_price'] - df['manufacturing_cost']) / df['discounted_price']) * 100
    df['discount_amount'] = df['actual_price'] - df['discounted_price']
    
    # Log-transform of prices (helps normalize price distribution)
    for col in ['discounted_price', 'actual_price', 'manufacturing_cost']:
        df[f'log_{col}'] = np.log1p(df[col])  # log1p adds 1 before taking log to handle zeros
    
    # Create category-specific price percentiles
    categories = df['category'].unique()
    
    for category in categories:
        cat_mask = df['category'] == category
        cat_prices = df.loc[cat_mask, 'discounted_price']
        
        # Create price percentile ranking within category (0-100)
        df.loc[cat_mask, 'price_percentile'] = pd.qcut(
            cat_prices, 
            q=100, 
            labels=False, 
            duplicates='drop'
        )
        
        # Create price segment (budget, mid-range, premium)
        df.loc[cat_mask, 'price_segment_numeric'] = pd.qcut(
            cat_prices,
            q=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2],  # 0=budget, 1=mid-range, 2=premium
            duplicates='drop'
        )
        
        # Map numeric segments to categorical
        segment_map = {0: 'budget', 1: 'mid-range', 2: 'premium'}
        df.loc[cat_mask, 'price_segment'] = df.loc[cat_mask, 'price_segment_numeric'].map(segment_map)
        
        # Calculate relative price (how much above/below category median)
        cat_median = cat_prices.median()
        df.loc[cat_mask, 'price_relative_to_median'] = (df.loc[cat_mask, 'discounted_price'] / cat_median) - 1
        
    logger.info(f"Created price features: {', '.join(list(set(df.columns) - set(df.columns)))}")
    return df

def encode_categorical_features(df):
    """
    Properly encode categorical features
    """
    logger.info("Encoding categorical features")
    df_encoded = df.copy()
    
    # Store the original category for later use
    original_category = df_encoded['category'].copy()
    
    # Identify categorical columns to encode
    cat_cols = ['brand', 'price_segment', 'category', 'subcategory']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # Brand tier mapping
    if 'brand' in df.columns:
        try:
            # Create brand tiers based on average pricing
            brand_avg_price = df.groupby('brand')['discounted_price'].mean().reset_index()
            brand_avg_price['brand_tier_num'] = pd.qcut(
                brand_avg_price['discounted_price'], 
                q=3, 
                labels=[0, 1, 2],
                duplicates='drop'
            )
            
            # Map numeric brand tiers to category labels
            tier_map = {0: 'budget', 1: 'mid-tier', 2: 'premium'}
            brand_avg_price['brand_tier'] = brand_avg_price['brand_tier_num'].map(tier_map)
            
            # Merge brand tiers back to main dataset
            df_encoded = df_encoded.merge(
                brand_avg_price[['brand', 'brand_tier']], 
                on='brand', 
                how='left'
            )
            
            # Add brand_tier to categorical columns to encode
            cat_cols.append('brand_tier')
            logger.info(f"Created brand tiers with {len(brand_avg_price)} brands")
        except Exception as e:
            logger.warning(f"Error creating brand tiers: {str(e)}")
    
    # Properly encode categorical columns with one-hot encoding
    # First convert to category dtype
    for col in cat_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype('category')
    
    # Get list of columns that actually exist in the dataframe
    cols_to_encode = [col for col in cat_cols if col in df_encoded.columns]
    
    # Apply one-hot encoding to all categorical columns
    if cols_to_encode:
        df_encoded = pd.get_dummies(
            df_encoded, 
            columns=cols_to_encode, 
            drop_first=False,  # Keep all categories
            prefix_sep='_',
            dummy_na=False
        )
        
        logger.info(f"One-hot encoded {len(cols_to_encode)} categorical columns")
    else:
        logger.warning("No categorical columns to encode")
    
    # Add back the original category for splitting purposes
    df_encoded['original_category'] = original_category
    
    return df_encoded

def split_dataset_by_category(df):
    """
    Split dataset into train/test sets for each category individually,
    ensuring balanced category representation
    """
    logger.info("Splitting dataset by category")
    train_dfs = []
    test_dfs = []
    
    # Use the original_category column that we preserved
    if 'original_category' not in df.columns:
        logger.error("No category column found for splitting")
        raise ValueError("Missing category column for splitting")
        
    categories = df['original_category'].unique()
    
    for category in categories:
        category_df = df[df['original_category'] == category].copy()
        logger.info(f"Splitting {category}: {len(category_df)} products")
        
        # Use stratified sampling based on price segments if available
        if 'price_segment_numeric' in category_df.columns:
            strat_col = 'price_segment_numeric'
        else:
            strat_col = None
        
        # Split the data
        train_df, test_df = train_test_split(
            category_df,
            test_size=0.2,
            random_state=42,
            stratify=category_df[strat_col] if strat_col else None
        )
        
        # Add to collections
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        
        logger.info(f"Split {category}: {len(train_df)} train, {len(test_df)} test")
    
    # Combine all categories
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    logger.info(f"Final split: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df

def visualize_price_distributions(df, outlier_stats):
    """
    Visualize price distributions before and after outlier handling
    """
    logger.info("Creating price distribution visualizations")
    
    categories = df['category'].unique()
    price_cols = ['discounted_price', 'actual_price', 'manufacturing_cost']
    
    for category in categories:
        category_data = df[df['category'] == category]
        
        for col in price_cols:
            if col in category_data.columns:
                plt.figure(figsize=(12, 6))
                
                # Create subplot for original distribution
                plt.subplot(1, 2, 1)
                sns.histplot(category_data[col], kde=True)
                
                # Add lines for outlier bounds
                if category in outlier_stats and col in outlier_stats[category]:
                    lower_bound = outlier_stats[category][col]['lower_bound']
                    upper_bound = outlier_stats[category][col]['upper_bound']
                    plt.axvline(x=lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
                    plt.axvline(x=upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
                
                plt.title(f"{category} - {col} Distribution")
                plt.legend()
                
                # Create subplot for log-transformed distribution
                plt.subplot(1, 2, 2)
                sns.histplot(np.log1p(category_data[col]), kde=True)
                plt.title(f"{category} - Log({col}) Distribution")
                
                plt.tight_layout()
                plt.savefig(f"visualizations/distributions/{category}_{col}_distribution.png")
                plt.close()
                
    logger.info("Price distribution visualizations complete")

def main():
    """Main function to run the data preparation process"""
    try:
        logger.info("Starting improved data preparation process")
        
        # Load the dataset
        df = load_dataset()
        logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Perform basic cleaning
        df_clean = clean_dataset(df)
        logger.info(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        
        # Save cleaned dataset
        df_clean.to_csv('data/cleaned/cleaned_dataset.csv', index=False)
        logger.info("Saved cleaned dataset")
        
        # Handle outliers with capping
        df_processed, outlier_stats = handle_outliers(df_clean)
        logger.info(f"Processed dataset with outlier handling: {df_processed.shape[0]} rows")
        
        # Create advanced price features
        df_processed = create_price_features(df_processed)
        logger.info(f"Added price features: {df_processed.shape[1]} total columns")
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df_processed)
        logger.info(f"Encoded dataset: {df_encoded.shape[1]} columns after encoding")
        
        # Visualize distributions
        visualize_price_distributions(df_processed, outlier_stats)
        
        # Split the dataset by category
        train_df, test_df = split_dataset_by_category(df_encoded)
        
        # Save processed datasets
        train_df.to_csv('data/processed/train_data.csv', index=False)
        test_df.to_csv('data/processed/test_data.csv', index=False)
        
        # Save category-specific datasets
        categories = df_encoded['original_category'].unique()
        for category in categories:
            # Create category-specific directory
            os.makedirs(f'data/processed/{category}', exist_ok=True)
            
            # Save category-specific train and test sets
            cat_train = train_df[train_df['original_category'] == category]
            cat_test = test_df[test_df['original_category'] == category]
            
            cat_train.to_csv(f'data/processed/{category}/train.csv', index=False)
            cat_test.to_csv(f'data/processed/{category}/test.csv', index=False)
            
            logger.info(f"Saved {category} data: {len(cat_train)} train, {len(cat_test)} test")
        
        logger.info("Improved data preparation process complete!")
        return df_encoded, train_df, test_df
        
    except Exception as e:
        logger.error(f"Error in data preparation process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 