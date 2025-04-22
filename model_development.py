import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
import traceback
import gc
import sys
import shutil

# Check if required packages are installed
try:
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import RandomizedSearchCV
except ImportError as e:
    print(f"Error: Required dependency not found - {e}")
    print("Please install required packages with: pip install xgboost scikit-learn matplotlib seaborn")
    sys.exit(1)

# Suppress warnings but keep error messages
warnings.filterwarnings('ignore')

# Function to check available memory
def check_memory():
    """Check if sufficient memory is available"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 * 1024 * 1024)
        total_gb = memory.total / (1024 * 1024 * 1024)
        percent_available = memory.available / memory.total * 100
        
        print(f"Memory Status: {available_gb:.1f}GB available out of {total_gb:.1f}GB total ({percent_available:.1f}%)")
        
        if percent_available < 20:
            print("Warning: Low memory available. Performance may be affected.")
            return False
        return True
    except ImportError:
        print("Warning: psutil not installed. Cannot check memory status.")
        return True
    except Exception as e:
        print(f"Warning: Failed to check memory - {e}")
        return True

# Function to verify output directories exist and are writable
def verify_directories():
    """Ensure output directories exist and are writable"""
    directories = [
        'models',
        'models/category_models',
        'visualizations/model_performance',
        'visualizations/feature_importance'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Test if directory is writable
            test_file = os.path.join(directory, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            print(f"Error: No permission to write to directory {directory}")
            return False
        except Exception as e:
            print(f"Error creating/accessing directory {directory}: {e}")
            return False
    
    return True

# Function to load the feature-engineered data for each category
def load_category_data():
    """
    Load the feature-engineered training and test datasets for each category
    """
    try:
        train_files = glob.glob('data/feature_engineered/*_train_features.csv')
        if not train_files:
            print("Error: No training files found. Run feature engineering first.")
            return {}
            
        categories = [os.path.basename(f).replace('_train_features.csv', '') for f in train_files]
        
        category_data = {}
        for category in categories:
            train_path = f'data/feature_engineered/{category}_train_features.csv'
            test_path = f'data/feature_engineered/{category}_test_features.csv'
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                try:
                    # Check file sizes to avoid loading very large files
                    train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
                    test_size_mb = os.path.getsize(test_path) / (1024 * 1024)
                    
                    if train_size_mb > 1000 or test_size_mb > 1000:  # If files are over 1GB
                        print(f"Warning: Files for {category} are very large. Skipping to prevent memory issues.")
                        continue
                    
                    # Load data
                    train_df = pd.read_csv(train_path)
                    test_df = pd.read_csv(test_path)
                    
                    # Check if required target column exists
                    if 'discounted_price' not in train_df.columns or 'discounted_price' not in test_df.columns:
                        print(f"Error: Target column 'discounted_price' not found in {category} data")
                        continue
                    
                    # Get columns to drop - only drop if they exist
                    cols_to_drop = []
                    cols_to_encode = ['brand_tier', 'product_type', 'price_segment']
                    
                    # Handle categorical columns by converting to numeric
                    for col in cols_to_encode:
                        if col in train_df.columns:
                            # Convert categorical columns using one-hot encoding
                            print(f"Converting categorical column: {col}")
                            # Use pandas get_dummies for one-hot encoding
                            train_dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
                            test_dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=True)
                            
                            # Make sure test has same columns as train
                            for col_name in train_dummies.columns:
                                if col_name not in test_dummies.columns:
                                    test_dummies[col_name] = 0
                            
                            # Make sure train has same columns as test
                            for col_name in test_dummies.columns:
                                if col_name not in train_dummies.columns:
                                    train_dummies[col_name] = 0
                            
                            # Add encoded columns
                            train_df = pd.concat([train_df, train_dummies], axis=1)
                            test_df = pd.concat([test_df, test_dummies], axis=1)
                            
                            # Add original column to drop list
                            cols_to_drop.append(col)
                    
                    for col in ['discounted_price', 'product_name', 'brand', 'subcategory', 'category', 'search_term', 'main_category']:
                        if col in train_df.columns and col not in cols_to_drop:
                            cols_to_drop.append(col)
                    
                    # Prepare features and target
                    X_train = train_df.drop(cols_to_drop, axis=1, errors='ignore')
                    y_train = train_df['discounted_price']
                    
                    X_test = test_df.drop(cols_to_drop, axis=1, errors='ignore')
                    y_test = test_df['discounted_price']
                    
                    # Check for any remaining non-numeric columns
                    non_numeric_cols_train = X_train.select_dtypes(exclude=['number']).columns.tolist()
                    if non_numeric_cols_train:
                        print(f"Warning: Dropping non-numeric columns in {category}: {non_numeric_cols_train}")
                        X_train = X_train.select_dtypes(include=['number'])
                        X_test = X_test.select_dtypes(include=['number'])
                    
                    # Verify data is not empty
                    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                        print(f"Error: Empty dataset for {category} after preprocessing")
                        continue
                    
                    # Store in dictionary
                    category_data[category] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                        'train_df': train_df,
                        'test_df': test_df
                    }
                    
                    print(f"Loaded {category}: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples, {X_train.shape[1]} features")
                except pd.errors.EmptyDataError:
                    print(f"Error: Empty file for {category}")
                except Exception as e:
                    print(f"Error loading data for {category}: {str(e)}")
                    traceback.print_exc()
        
        if not category_data:
            print("Error: No valid data could be loaded for any category")
        
        return category_data
    except Exception as e:
        print(f"Critical error in data loading: {str(e)}")
        traceback.print_exc()
        return {}

# Function to train a model for a specific category
def train_category_model(category, X_train, y_train, X_test, y_test, hyperparameter_tuning=True):
    """
    Train an XGBoost model for a specific category with optional hyperparameter tuning
    """
    try:
        print(f"\nTraining model for: {category}")
        
        # Check for NaN values
        if X_train.isnull().any().any() or X_test.isnull().any().any() or y_train.isnull().any() or y_test.isnull().any():
            print("Warning: Data contains NaN values. Filling with appropriate values.")
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_test.median())
            y_train = y_train.fillna(y_train.median())
            y_test = y_test.fillna(y_test.median())
        
        # Basic data validation
        if X_train.shape[0] < 10 or X_test.shape[0] < 5:
            print(f"Error: Too few samples for {category}. Need at least 10 training and 5 test samples.")
            return None
            
        # Check for price range consistency
        y_min, y_max = y_train.min(), y_train.max()
        price_range = y_max - y_min
        if price_range <= 0:
            print(f"Error: No price variation in {category}. All prices are {y_min}.")
            return None
            
        # Check for extreme outliers
        q1, q3 = np.percentile(y_train, [25, 75])
        iqr = q3 - q1
        extreme_outliers = ((y_train < (q1 - 3 * iqr)) | (y_train > (q3 + 3 * iqr))).sum()
        if extreme_outliers > 0:
            print(f"Warning: Found {extreme_outliers} extreme price outliers in {category}")
        
        # Define base parameters
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'n_estimators': 100  # Default value
        }
        
        # Determine if we should do hyperparameter tuning based on dataset size
        # Skip for very large datasets to prevent memory issues
        large_dataset = X_train.shape[0] > 10000 or X_train.shape[1] > 100
        if large_dataset and hyperparameter_tuning:
            print("Warning: Large dataset detected. Using limited hyperparameter tuning.")
            limited_tuning = True
        else:
            limited_tuning = False
        
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid for tuning
            if limited_tuning:
                # Smaller grid for large datasets
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
                n_iter = 5
                cv = 2
            else:
                # Full grid for normal datasets
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2]
                }
                n_iter = 10
                cv = 3
            
            try:
                # Use RandomizedSearchCV for efficiency
                xgb_model = xgb.XGBRegressor(**base_params)
                random_search = RandomizedSearchCV(
                    xgb_model, 
                    param_distributions=param_grid,
                    n_iter=n_iter,  # Number of parameter settings sampled
                    scoring='neg_mean_squared_error',
                    cv=cv,  # Number of cross-validation folds
                    verbose=1,
                    n_jobs=-1,  # Use all available cores
                    random_state=42,
                    error_score='raise'  # Raise error if a model fails
                )
                
                # Fit the model
                random_search.fit(X_train, y_train)
                
                # Get best parameters
                best_params = random_search.best_params_
                print(f"Best parameters: {best_params}")
                
                # Update base parameters with best parameters
                base_params.update(best_params)
                
            except Exception as e:
                print(f"Hyperparameter tuning failed: {str(e)}")
                print("Falling back to default parameters")
                # Don't update base_params, fallback to defaults
            
            # Force garbage collection after hyperparameter tuning
            gc.collect()
        
        # Train final model with best parameters
        final_model = xgb.XGBRegressor(**base_params)
            
        # Train the final model
        final_model.fit(X_train, y_train)
        
        # Manual evaluation approach (replacement for early stopping)
        print("Evaluating model performance...")
        
        # Create separate evaluation function to check model performance
        def evaluate_model(model, X, y, dataset_name):
            try:
                preds = model.predict(X)
                
                # Handle extreme predictions by clipping to reasonable ranges
                # Clip to min/max of training data with a small buffer
                buffer_factor = 0.1  # Allow predictions to be 10% outside the range of training data
                min_price = max(0, y_train.min() * (1 - buffer_factor))  # Don't allow negative prices
                max_price = y_train.max() * (1 + buffer_factor)
                
                # Clip predictions to valid range
                preds_clipped = np.clip(preds, min_price, max_price)
                
                # Check if clipping changed a lot of values
                clipped_count = np.sum((preds != preds_clipped))
                if clipped_count > 0:
                    clipped_percent = (clipped_count / len(preds)) * 100
                    print(f"Warning: {clipped_count} predictions ({clipped_percent:.1f}%) were outside valid price range and were clipped")
                
                # Calculate metrics with clipped predictions
                rmse = np.sqrt(mean_squared_error(y, preds_clipped))
                mae = mean_absolute_error(y, preds_clipped)
                r2 = r2_score(y, preds_clipped)
                
                # Check if metrics are valid
                if np.isnan(rmse) or np.isnan(mae) or np.isnan(r2):
                    print(f"Warning: Invalid metrics for {dataset_name}")
                    return np.inf, np.inf, -np.inf, preds_clipped
                    
                print(f"{dataset_name} RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
                
                # Evaluate model quality
                if r2 < 0:
                    print(f"Warning: Negative R² for {dataset_name}. Model performs worse than a constant predictor.")
                elif r2 < 0.3:
                    print(f"Warning: Low R² for {dataset_name}. Model has poor predictive power.")
                elif r2 > 0.95 and len(X) > 30:
                    print(f"Warning: Suspiciously high R² for {dataset_name}. Model may be overfitting.")
                
                return rmse, mae, r2, preds_clipped
            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                return np.inf, np.inf, -np.inf, np.zeros(len(y))
        
        # Evaluate on train and test
        train_metrics = evaluate_model(final_model, X_train, y_train, "Training")
        test_metrics = evaluate_model(final_model, X_test, y_test, "Test")
        
        # Quality check: Test R² should not be dramatically worse than training R²
        if train_metrics[2] > 0.7 and test_metrics[2] < 0.2 and len(X_test) > 30:
            print("Warning: Large gap between training and test performance. Model is likely overfitting.")
        
        # Verify model is not just predicting the mean
        y_mean = np.mean(y_train)
        mean_baseline_rmse = np.sqrt(np.mean((y_test - y_mean) ** 2))
        
        if test_metrics[0] >= mean_baseline_rmse * 0.95:
            print(f"Warning: Model performance is not better than predicting the mean (RMSE: {mean_baseline_rmse:.2f}).")
            print("Model may not have learned useful patterns.")
        
        print(f"Model trained with {final_model.n_estimators} trees")
        
        # Get feature importance
        feature_importance = final_model.feature_importances_
        feature_names = X_train.columns
        
        # Create DataFrame for feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        # Evaluate model - reuse the computed metrics
        train_rmse, train_mae, train_r2, train_preds = train_metrics
        test_rmse, test_mae, test_r2, test_preds = test_metrics
        
        # Check if model quality is acceptable
        model_quality = "good"
        if test_r2 < 0.3 or np.isnan(test_r2) or test_r2 == -np.inf:
            model_quality = "poor"
            print("Warning: Model quality is poor based on test performance.")
        
        # Create dictionary with model and evaluation metrics
        model_results = {
            'model': final_model,
            'parameters': final_model.get_params(),
            'feature_importance': feature_importance_df,
            'metrics': {
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'model_quality': model_quality
            },
            'predictions': {
                'train_preds': train_preds,
                'test_preds': test_preds
            }
        }
        
        return model_results
    except Exception as e:
        print(f"Error in model training for {category}: {str(e)}")
        traceback.print_exc()
        return None

# Function to visualize model performance
def visualize_model_performance(category, model_results, X_test, y_test):
    """
    Create visualizations for model performance and feature importance
    """
    try:
        if model_results is None:
            print(f"Skipping visualization for {category} - no model results")
            return
            
        model = model_results['model']
        metrics = model_results['metrics']
        feature_importance_df = model_results['feature_importance']
        
        # Predict test values
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Error making predictions for {category}: {str(e)}")
            return
        
        # Handle NaN values in predictions or actual values
        mask = ~(np.isnan(y_test) | np.isnan(y_pred))
        if not mask.any():
            print(f"Error: All predictions or actual values are NaN for {category}")
            return
            
        y_test_clean = y_test[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_test_clean) == 0:
            print(f"Error: No valid predictions for {category}")
            return
        
        # 1. Actual vs. Predicted Plot
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_clean, y_pred_clean, alpha=0.5)
            
            # Only draw line if we have valid min/max
            if not (np.isnan(y_test_clean.min()) or np.isnan(y_test_clean.max())):
                plt.plot([y_test_clean.min(), y_test_clean.max()], [y_test_clean.min(), y_test_clean.max()], 'r--')
                
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title(f'{category} - Actual vs. Predicted Price\nTest RMSE: {metrics["test_rmse"]:.2f}, R²: {metrics["test_r2"]:.3f}')
            plt.tight_layout()
            plt.savefig(f'visualizations/model_performance/{category}_actual_vs_predicted.png')
            plt.close()
        except Exception as e:
            print(f"Error creating actual vs predicted plot for {category}: {str(e)}")
        
        # 2. Feature Importance Plot
        try:
            if len(feature_importance_df) > 0:
                plt.figure(figsize=(12, 8))
                top_n = min(15, len(feature_importance_df))  # In case we have fewer than 15 features
                top_features = feature_importance_df.head(top_n)
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title(f'{category} - Top {top_n} Feature Importance')
                plt.tight_layout()
                plt.savefig(f'visualizations/feature_importance/{category}_feature_importance.png')
                plt.close()
        except Exception as e:
            print(f"Error creating feature importance plot for {category}: {str(e)}")
        
        # 3. Residual Plot
        try:
            residuals = y_test_clean - y_pred_clean
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred_clean, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Price')
            plt.ylabel('Residuals')
            plt.title(f'{category} - Residual Plot\nMAE: {metrics["test_mae"]:.2f}')
            plt.tight_layout()
            plt.savefig(f'visualizations/model_performance/{category}_residuals.png')
            plt.close()
        except Exception as e:
            print(f"Error creating residual plot for {category}: {str(e)}")
        
        # 4. Error Distribution Plot
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title(f'{category} - Error Distribution')
            plt.tight_layout()
            plt.savefig(f'visualizations/model_performance/{category}_error_distribution.png')
            plt.close()
        except Exception as e:
            print(f"Error creating error distribution plot for {category}: {str(e)}")
            
    except Exception as e:
        print(f"Error in visualization for {category}: {str(e)}")
        traceback.print_exc()

# Function to save model and results
def save_model_results(category, model_results):
    """
    Save the trained model and results
    """
    try:
        if model_results is None:
            print(f"Skipping saving results for {category} - no model results")
            return
            
        # Save the model
        model_path = f'models/category_models/{category}_model.pkl'
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_results['model'], f)
        except Exception as e:
            print(f"Error saving model for {category}: {str(e)}")
            
            # Try saving with a different approach if standard pickle fails
            try:
                # Make a copy of the model results without the model itself
                results_copy = {k: v for k, v in model_results.items() if k != 'model'}
                
                # Save the model using joblib if available
                try:
                    from sklearn.externals import joblib
                except ImportError:
                    import joblib
                    
                joblib.dump(model_results['model'], model_path)
                print(f"Model saved using joblib for {category}")
            except:
                print(f"Failed to save model for {category} using alternative methods")
        
        # Save feature importance
        try:
            feature_importance_path = f'models/category_models/{category}_feature_importance.csv'
            model_results['feature_importance'].to_csv(feature_importance_path, index=False)
        except Exception as e:
            print(f"Error saving feature importance for {category}: {str(e)}")
        
        # Save metrics
        try:
            metrics_path = f'models/category_models/{category}_metrics.json'
            with open(metrics_path, 'w') as f:
                # Ensure all values are JSON serializable
                metrics_copy = {}
                for k, v in model_results['metrics'].items():
                    if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                        metrics_copy[k] = float(v)
                    else:
                        metrics_copy[k] = v
                        
                json.dump(metrics_copy, f, indent=4)
        except Exception as e:
            print(f"Error saving metrics for {category}: {str(e)}")
        
        # Save parameters
        try:
            params_path = f'models/category_models/{category}_parameters.json'
            with open(params_path, 'w') as f:
                params = {}
                for k, v in model_results['parameters'].items():
                    if isinstance(v, (np.ndarray, list)):
                        params[k] = str(v)
                    elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                        params[k] = float(v)
                    else:
                        params[k] = v
                json.dump(params, f, indent=4)
        except Exception as e:
            print(f"Error saving parameters for {category}: {str(e)}")
        
        print(f"Model and results saved for {category}")
    except Exception as e:
        print(f"Critical error saving model results for {category}: {str(e)}")
        traceback.print_exc()

# Main function to run the model development process
def main(hyperparameter_tuning=True):
    print("Starting XGBoost model development...")
    
    # Check if the environment is ready
    if not verify_directories():
        print("Error: Failed to verify output directories")
        return None
        
    # Check memory
    check_memory()
    
    # Load feature-engineered data for each category
    category_data = load_category_data()
    if not category_data:
        print("Error: No data loaded. Exiting.")
        return None
    
    # Dictionary to store model results
    all_model_results = {}
    categories_with_errors = []
    
    # Train models for each category
    for category, data in category_data.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing category: {category}")
            print(f"{'='*50}")
            
            # Create a backup of the data just in case
            try:
                backup_path = f'data/feature_engineered/{category}_backup'
                os.makedirs(backup_path, exist_ok=True)
                data['train_df'].to_csv(f'{backup_path}/train_backup.csv', index=False)
                data['test_df'].to_csv(f'{backup_path}/test_backup.csv', index=False)
            except:
                print("Warning: Failed to create data backup - continuing anyway")
            
            # Train the model
            model_results = train_category_model(
                category,
                data['X_train'],
                data['y_train'],
                data['X_test'],
                data['y_test'],
                hyperparameter_tuning
            )
            
            # Force garbage collection
            gc.collect()
            
            if model_results is not None:
                # Visualize model performance
                visualize_model_performance(
                    category,
                    model_results,
                    data['X_test'],
                    data['y_test']
                )
                
                # Save model and results
                save_model_results(category, model_results)
                
                # Store in dictionary
                all_model_results[category] = model_results
            else:
                categories_with_errors.append(category)
            
        except Exception as e:
            print(f"Critical error processing category {category}: {str(e)}")
            traceback.print_exc()
            categories_with_errors.append(category)
            continue
            
        # Force garbage collection after each category
        gc.collect()
    
    # Create summary of model performance
    print("\nModel Performance Summary:")
    
    if not all_model_results:
        print("No models were successfully trained.")
        return None
        
    try:
        performance_data = []
        for category, results in all_model_results.items():
            metrics = results['metrics']
            performance_data.append({
                'Category': category,
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Test R²': metrics['test_r2'],
                'Quality': metrics.get('model_quality', 'unknown')
            })
        
        # Create DataFrame and save to CSV
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv('models/model_performance_summary.csv', index=False)
        
        # Print summary
        print(performance_df.to_string(index=False))
        
        if categories_with_errors:
            print(f"\nCategories with errors: {', '.join(categories_with_errors)}")
        
        # Count good and poor models
        if 'Quality' in performance_df.columns:
            good_models = (performance_df['Quality'] == 'good').sum()
            poor_models = (performance_df['Quality'] == 'poor').sum()
            
            print(f"\nModel quality summary:")
            print(f"- Good models: {good_models} out of {len(performance_df)}")
            print(f"- Poor models: {poor_models} out of {len(performance_df)}")
            
            # Overall project status
            if good_models >= len(performance_df) * 0.7:
                print("\nPROJECT STATUS: SUCCESSFUL - Most models have good quality")
            elif good_models >= len(performance_df) * 0.5:
                print("\nPROJECT STATUS: MIXED RESULTS - Some models need improvement")
            else:
                print("\nPROJECT STATUS: NEEDS REVISION - Most models have poor quality")
                
    except Exception as e:
        print(f"Error creating performance summary: {str(e)}")
    
    return all_model_results

if __name__ == "__main__":
    # Check for command-line args
    import argparse
    parser = argparse.ArgumentParser(description='Train XGBoost models for price prediction')
    parser.add_argument('--quick', action='store_true', help='Run without hyperparameter tuning for quick results')
    parser.add_argument('--test', action='store_true', help='Run on a single category for testing')
    args = parser.parse_args()
    
    try:
        # Set hyperparameter_tuning to False for faster development/testing
        if args.quick:
            print("Running in QUICK mode (no hyperparameter tuning)")
            main(hyperparameter_tuning=False)
        elif args.test:
            print("Running in TEST mode (single category)")
            # Load just one category and train
            category_data = load_category_data()
            if category_data:
                category = list(category_data.keys())[0]
                data = category_data[category]
                
                print(f"Testing with category: {category}")
                model_results = train_category_model(
                    category,
                    data['X_train'],
                    data['y_train'],
                    data['X_test'],
                    data['y_test'],
                    hyperparameter_tuning=False
                )
                
                if model_results:
                    visualize_model_performance(category, model_results, data['X_test'], data['y_test'])
                    save_model_results(category, model_results)
        else:
            main(hyperparameter_tuning=True)
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        traceback.print_exc() 