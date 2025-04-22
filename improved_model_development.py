import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
import time
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_development.log'),
        logging.StreamHandler()
    ]
)

# Suppress warnings but keep error messages
warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('models/improved', exist_ok=True)
os.makedirs('models/improved/category_models', exist_ok=True)
os.makedirs('visualizations/improved/model_performance', exist_ok=True)
os.makedirs('visualizations/improved/feature_importance', exist_ok=True)

class ImprovedModelDevelopment:
    """Class for developing improved XGBoost models for pricing prediction."""
    
    def __init__(self, dataset_dir='data/engineered'):
        """Initialize the model development process.
        
        Args:
            dataset_dir: Directory containing engineered datasets
        """
        self.dataset_dir = dataset_dir
        self.outlier_stats = {}
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
        # Check if outlier stats file exists, create if not
        self.outlier_stats_path = 'logs/outlier_stats.json'
        if not os.path.exists(self.outlier_stats_path):
            logging.info("No outlier stats file found. Will create one during training.")
            self.outlier_stats = {}
        else:
            try:
                with open(self.outlier_stats_path, 'r') as f:
                    self.outlier_stats = json.load(f)
                logging.info(f"Loaded outlier stats for {len(self.outlier_stats)} categories")
            except Exception as e:
                logging.error(f"Error loading outlier stats: {str(e)}")
                self.outlier_stats = {}
    
    def load_category_data(self, category):
        """Load data for a specific category.
        
        Args:
            category: The product category to load data for
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or None if loading fails
        """
        try:
            # Load train data
            train_file = os.path.join(self.dataset_dir, f"{category}_train.csv")
            if not os.path.exists(train_file):
                logging.error(f"Training file not found: {train_file}")
                return None
            
            train_data = pd.read_csv(train_file)
            
            # Load test data
            test_file = os.path.join(self.dataset_dir, f"{category}_test.csv")
            if not os.path.exists(test_file):
                logging.error(f"Test file not found: {test_file}")
                return None
            
            test_data = pd.read_csv(test_file)
            
            # Ensure 'price' column exists
            if 'price' not in train_data.columns or 'price' not in test_data.columns:
                logging.error(f"Price column not found in data for category {category}")
                return None
            
            # Split features and target
            y_train = train_data['price']
            X_train = train_data.drop('price', axis=1)
            
            y_test = test_data['price']
            X_test = test_data.drop('price', axis=1)
            
            # Filter out non-numeric columns
            numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
            X_train = X_train[numeric_cols]
            X_test = X_test[numeric_cols]
            
            # Ensure data is not empty
            if len(X_train) < 10 or len(X_test) < 5:
                logging.warning(f"Not enough data for category {category}. Train: {len(X_train)}, Test: {len(X_test)}")
                return None
            
            # Calculate and save outlier statistics
            self._calculate_outlier_stats(category, y_train)
            
            # Filter extreme outliers if there are enough samples
            if len(y_train) >= 50:
                # Get outlier boundaries
                lower_bound = self.outlier_stats[category]['lower_bound']
                upper_bound = self.outlier_stats[category]['upper_bound']
                
                # Log how many outliers we're removing
                outliers_count = sum((y_train < lower_bound) | (y_train > upper_bound))
                if outliers_count > 0:
                    logging.info(f"Removing {outliers_count} outliers from {category} training data")
                
                # Filter outliers from training data
                outlier_mask = (y_train >= lower_bound) & (y_train <= upper_bound)
                X_train = X_train[outlier_mask]
                y_train = y_train[outlier_mask]
            
            logging.info(f"Loaded data for category {category}. Train: {len(X_train)}, Test: {len(X_test)}")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error loading data for category {category}: {str(e)}")
            return None
    
    def _calculate_outlier_stats(self, category, prices):
        """Calculate outlier statistics for a category.
        
        Args:
            category: The product category
            prices: Series of prices
        """
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        
        # Calculate outlier boundaries
        lower_bound = max(0, q1 - 1.5 * iqr)  # Ensure we don't go below 0
        upper_bound = q3 + 1.5 * iqr
        
        # Additional statistics for pricing strategy
        min_price = prices.min()
        max_price = prices.max()
        median_price = prices.median()
        
        # Store statistics
        self.outlier_stats[category] = {
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'min_price': float(min_price),
            'max_price': float(max_price),
            'median_price': float(median_price)
        }
        
        # Save to file
        with open(self.outlier_stats_path, 'w') as f:
            json.dump(self.outlier_stats, f, indent=4)
    
    def train_model(self, category, X_train, X_test, y_train, y_test):
        """Train XGBoost model with hyperparameter tuning for a category.
        
        Args:
            category: The product category
            X_train, X_test, y_train, y_test: Training and testing data
        
        Returns:
            Trained model or None if training fails
        """
        try:
            logging.info(f"Training model for category {category}")
            
            # Scale features for better model performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler for later use
            self.scalers[category] = scaler
            
            # Define hyperparameter search space
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0.1, 0.5, 1, 5]
            }
            
            # Initialize model
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            # Use cross-validation to avoid overfitting
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # RandomizedSearchCV with MSE as scoring metric
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                scoring='neg_mean_squared_error',
                cv=cv,
                n_iter=20,  # Number of parameter settings sampled
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
            
            # Train the model
            search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = search.best_estimator_
            logging.info(f"Best parameters for {category}: {search.best_params_}")
            
            # Evaluate model
            metrics = self._evaluate_model(category, best_model, X_train_scaled, X_test_scaled, 
                                          y_train, y_test)
            self.metrics[category] = metrics
            
            # Visualize results
            self._visualize_model_performance(category, best_model, X_test_scaled, y_test, metrics)
            
            # Store model
            self.models[category] = best_model
            
            # Save model to disk
            self._save_model(category, best_model)
            
            return best_model
        
        except Exception as e:
            logging.error(f"Error training model for category {category}: {str(e)}")
            return None
    
    def _evaluate_model(self, category, model, X_train_scaled, X_test_scaled, y_train, y_test):
        """Evaluate model performance on training and test data.
        
        Args:
            category: The product category
            model: Trained XGBoost model
            X_train_scaled, X_test_scaled: Scaled feature data
            y_train, y_test: Target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train))
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test))
        
        # Calculate % of predictions within 10% of actual price
        within_10pct_train = np.mean(np.abs((y_train - y_train_pred) / y_train) <= 0.1)
        within_10pct_test = np.mean(np.abs((y_test - y_test_pred) / y_test) <= 0.1)
        
        # Calculate % of predictions within 20% of actual price
        within_20pct_train = np.mean(np.abs((y_train - y_train_pred) / y_train) <= 0.2)
        within_20pct_test = np.mean(np.abs((y_test - y_test_pred) / y_test) <= 0.2)
        
        # Log metrics
        logging.info(f"Evaluation metrics for {category}:")
        logging.info(f"  Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        logging.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logging.info(f"  Test MAPE: {test_mape:.4f}")
        logging.info(f"  % within 10% on test: {within_10pct_test:.2%}")
        
        # Create metrics dictionary
        metrics = {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mape': float(train_mape),
            'test_mape': float(test_mape),
            'train_within_10pct': float(within_10pct_train),
            'test_within_10pct': float(within_10pct_test),
            'train_within_20pct': float(within_20pct_train),
            'test_within_20pct': float(within_20pct_test)
        }
        
        return metrics
    
    def _visualize_model_performance(self, category, model, X_test_scaled, y_test, metrics):
        """Create visualizations for model performance.
        
        Args:
            category: The product category
            model: Trained XGBoost model
            X_test_scaled: Scaled test features
            y_test: Test target values
            metrics: Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance for {category}', fontsize=16)
        
        # 1. Scatter plot of predicted vs actual prices
        ax = axes[0, 0]
        ax.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Predicted vs Actual Prices')
        
        # Add metrics text
        text = f"RMSE: ${metrics['test_rmse']:.2f}\n"
        text += f"R²: {metrics['test_r2']:.2f}\n"
        text += f"MAPE: {metrics['test_mape']:.2%}\n"
        text += f"Within 10%: {metrics['test_within_10pct']:.2%}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals plot
        ax = axes[0, 1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Price')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals Plot')
        
        # 3. Histogram of residuals
        ax = axes[1, 0]
        ax.hist(residuals, bins=20, alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Residual Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        
        # 4. Feature importance
        ax = axes[1, 1]
        feature_importance = model.feature_importances_
        feature_names = X_test_scaled.shape[1]
        
        # Get feature names from the model
        features = [f'f{i}' for i in range(feature_names)]
        
        # Sort by importance
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title('Top 10 Feature Importance')
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
        figure_path = os.path.join('visualizations/improved/model_performance', f'{category}_model_performance.png')
        plt.savefig(figure_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved model performance visualization to {figure_path}")
    
    def _save_model(self, category, model):
        """Save model and metrics to disk.
        
        Args:
            category: The product category
            model: Trained XGBoost model
        """
        # Create directory if it doesn't exist
        os.makedirs('models/improved/category_models', exist_ok=True)
        
        # Save model
        model_path = os.path.join('models/improved/category_models', f'{category}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = os.path.join('models/improved/category_models', f'{category}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[category], f)
        
        # Save metrics
        metrics_path = os.path.join('models/improved/category_models', f'{category}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics[category], f, indent=4)
        
        logging.info(f"Saved model, scaler, and metrics for {category}")
    
    def run_training(self):
        """Run the training process for all categories."""
        # Get all available categories from engineered data files
        categories = set()
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith('_train.csv'):
                category = filename.replace('_train.csv', '')
                categories.add(category)
        
        logging.info(f"Found {len(categories)} categories to train models for: {', '.join(categories)}")
        
        for category in sorted(categories):
            logging.info(f"Processing category: {category}")
            
            # Load data for this category
            data = self.load_category_data(category)
            if data is None:
                logging.error(f"Skipping {category} due to data loading error")
                continue
            
            X_train, X_test, y_train, y_test = data
            
            # Train model
            model = self.train_model(category, X_train, X_test, y_train, y_test)
            if model is None:
                logging.error(f"Skipping {category} due to training error")
                continue
            
            logging.info(f"Successfully trained model for {category}")
        
        # Save combined metrics for all categories
        metrics_path = os.path.join('models/improved', 'all_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate summary of model performance
        self._generate_performance_summary()
        
        logging.info("Model training completed.")
    
    def _generate_performance_summary(self):
        """Generate a summary of model performance across all categories."""
        if not self.metrics:
            logging.warning("No metrics available to generate summary")
            return
        
        summary = {
            'average_test_rmse': np.mean([m['test_rmse'] for m in self.metrics.values()]),
            'average_test_r2': np.mean([m['test_r2'] for m in self.metrics.values()]),
            'average_test_mape': np.mean([m['test_mape'] for m in self.metrics.values()]),
            'average_within_10pct': np.mean([m['test_within_10pct'] for m in self.metrics.values()]),
            'best_category': '',
            'worst_category': '',
            'category_count': len(self.metrics)
        }
        
        # Find best and worst categories based on R2
        categories = list(self.metrics.keys())
        r2_values = [self.metrics[cat]['test_r2'] for cat in categories]
        
        best_idx = np.argmax(r2_values)
        worst_idx = np.argmin(r2_values)
        
        summary['best_category'] = {
            'name': categories[best_idx],
            'test_r2': self.metrics[categories[best_idx]]['test_r2'],
            'test_rmse': self.metrics[categories[best_idx]]['test_rmse'],
            'test_mape': self.metrics[categories[best_idx]]['test_mape']
        }
        
        summary['worst_category'] = {
            'name': categories[worst_idx],
            'test_r2': self.metrics[categories[worst_idx]]['test_r2'],
            'test_rmse': self.metrics[categories[worst_idx]]['test_rmse'],
            'test_mape': self.metrics[categories[worst_idx]]['test_mape']
        }
        
        # Save summary to file
        summary_path = os.path.join('models/improved', 'performance_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Log summary statistics
        logging.info("Model Performance Summary:")
        logging.info(f"  Average Test RMSE: ${summary['average_test_rmse']:.2f}")
        logging.info(f"  Average Test R²: {summary['average_test_r2']:.4f}")
        logging.info(f"  Average Test MAPE: {summary['average_test_mape']:.2%}")
        logging.info(f"  Average Within 10%: {summary['average_within_10pct']:.2%}")
        logging.info(f"  Best Category: {summary['best_category']['name']} (R²: {summary['best_category']['test_r2']:.4f})")
        logging.info(f"  Worst Category: {summary['worst_category']['name']} (R²: {summary['worst_category']['test_r2']:.4f})")

def main():
    """Main function to run the improved model development process."""
    logging.info("Starting improved model development process")
    
    # Create model development instance
    model_dev = ImprovedModelDevelopment()
    
    # Run training for all categories
    model_dev.run_training()
    
    logging.info("Improved model development completed")

if __name__ == "__main__":
    main() 