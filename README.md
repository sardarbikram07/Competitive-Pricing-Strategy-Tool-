# Competitive Pricing Strategy System for New Sellers

This system helps new sellers enter the market with competitive pricing strategies based on XGBoost models trained on product categories.

## System Overview

The Competitive Pricing Strategy System consists of several components:

1. **Data Preparation**: Processing raw data for model training
2. **Feature Engineering**: Creating category-specific features for price prediction
3. **Model Development**: Training XGBoost models for price prediction
4. **Pricing Strategy**: Implementing competitive pricing algorithms
5. **Streamlit Application**: User interface for pricing recommendations

## Project Structure

```
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed data files
│   └── engineered/                # Feature engineered data
├── models/                        # Trained models
│   ├── category_models/           # Original category models
│   └── improved/                  # Improved models with better performance
├── visualizations/                # Visualizations
│   ├── distributions/             # Price distributions
│   ├── model_performance/         # Model performance charts
│   ├── feature_importance/        # Feature importance visualizations
│   └── pricing_strategies/        # Pricing strategy visualizations
├── logs/                          # Log files
├── pricing_strategies/            # Saved pricing recommendations
├── data_preparation.py            # Data preparation script
├── feature_engineering_fixed.py   # Feature engineering script
├── model_development.py           # Original model development
├── improved_model_development.py  # Improved model development
├── pricing_strategy.py            # Pricing strategy implementation
├── streamlit_app.py               # Streamlit application
├── checklist.md                   # Project implementation checklist
└── README.md                      # This file
```

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

## Usage

### 1. Data Processing Pipeline

Run the data preparation and feature engineering scripts:

```bash
python data_preparation.py
python feature_engineering_fixed.py
```

### 2. Model Development

Train the improved XGBoost models:

```bash
python improved_model_development.py
```

### 3. Pricing Strategy

Test the pricing strategy implementation:

```bash
python pricing_strategy.py
```

### 4. Streamlit Application

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Features

### Data Preparation
- Data cleaning and validation
- Category segmentation
- Train/test splitting
- Price outlier handling

### Feature Engineering
- Category-specific pricing benchmarks
- Rating and discount correlations
- Profit margin calculations
- Competitive price factors

### Model Development
- XGBoost models for each product category
- Cross-validation and hyperparameter tuning
- Performance metrics (RMSE, MAE, MAPE, R²)
- Feature importance analysis

### Pricing Strategy
- Competitive pricing algorithms
- Price elasticity analysis
- Market saturation adaptation
- Brand strength considerations
- Profit margin protection

### Streamlit Application
- Single product pricing
- Batch pricing for multiple products
- Market analysis and visualization
- Pricing strategy history

## Pricing Strategy Algorithms

The system implements several pricing strategies based on market conditions:

1. **Aggressive Market Entry**: 25%+ discount for quick market entry
2. **Value-Oriented Strategy**: 15-25% discount balancing value and profit
3. **Competitive Positioning**: 5-15% discount for established products
4. **Premium Positioning**: <5% discount for premium products
5. **Deep Discount Strategy**: For highly saturated markets
6. **Thin-Margin Volume Strategy**: For price-sensitive categories

## Model Performance

The models are evaluated using multiple metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Within-10% Percentage

## Contributing

This project is maintained by Bikram Sardar and Alapan Sen. Contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
