import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pricing_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Create directories for outputs
os.makedirs('pricing_strategies', exist_ok=True)
os.makedirs('visualizations/pricing_strategies', exist_ok=True)

class PricingStrategy:
    """
    A class for implementing competitive pricing strategies
    based on the improved pricing models
    """
    
    def __init__(self, 
                models_dir='models/improved/category_models',
                aggressive_discount_range=(0.15, 0.25),  # 15-25% below market
                min_profit_margin=0.08,  # 8% minimum profit margin
                category_min_margins=None  # Category-specific minimum margins
                ):
        """
        Initialize the pricing strategy
        
        Parameters:
        -----------
        models_dir : str
            Directory containing the trained models
        aggressive_discount_range : tuple
            Range of discount to apply (min, max) as a percentage
        min_profit_margin : float
            Default minimum profit margin to ensure
        category_min_margins : dict, optional
            Dictionary mapping category names to their minimum profit margins
        """
        self.models_dir = models_dir
        self.aggressive_discount_range = aggressive_discount_range
        self.min_profit_margin = min_profit_margin
        self.models = {}
        self.metrics = {}
        self.category_benchmarks = {}
        
        # Define category-specific cost thresholds
        # Each tuple contains (warning_threshold, viability_threshold)
        self.category_thresholds = {
            'Smartwatches': (0.95, 1.15),       # More tolerant - can handle higher costs
            'Mobile Accessories': (0.9, 1.05),   # Moderate tolerance
            'Kitchen Appliances': (0.9, 1.05),   # Moderate tolerance
            'Cameras': (0.85, 1.0),              # Lower tolerance
            'Audio': (0.8, 0.95),                # Low tolerance
            'Computers': (0.85, 1.0),            # Lower tolerance
            'Home Entertainment': (0.8, 0.95),   # Low tolerance
            'Home Improvement': (0.8, 0.95),     # Low tolerance
            'Home Office': (0.8, 0.95),          # Low tolerance 
            'Climate Control': (0.85, 1.0)       # Lower tolerance
        }
        
        # Category-specific calibration factors
        self.calibration_factors = {
            'Audio': 1.25,                # Observed 0.63x -> adjust by 1.25 to get closer to real prices
            'Cameras': 1.2,               # Observed 0.65x
            'Climate Control': 1.2,       # Observed 0.64x
            'Computers': 1.2,             # Observed 0.64x
            'Home Entertainment': 1.45,   # Observed 0.53x
            'Home Improvement': 1.2,      # Observed 0.63x
            'Home Office': 1.45,          # Observed 0.54x
            'Kitchen Appliances': 1.1,    # Observed 0.71x
            'Mobile Accessories': 1.2,    # Observed 0.69x
            'Smartwatches': 1.05          # Observed 0.77x (most accurate)
        }
        
        # Default category-specific minimum margins
        self.category_min_margins = {
            'Smartwatches': 0.05,         # Can operate on thinner margins
            'Mobile Accessories': 0.07,    # Thin margins but not as thin as smartwatches
            'Kitchen Appliances': 0.08,    # Standard margin
            'Cameras': 0.10,               # Higher margin needed
            'Audio': 0.12,                 # Higher margin needed
            'Computers': 0.09,             # Slightly higher than standard
            'Home Entertainment': 0.12,    # Higher margin needed
            'Home Improvement': 0.10,      # Higher margin needed
            'Home Office': 0.10,           # Higher margin needed
            'Climate Control': 0.09        # Slightly higher than standard
        }
        
        # Override with any provided category-specific margins
        if category_min_margins:
            self.category_min_margins.update(category_min_margins)
        
        # Load models and metrics
        self._load_models_and_metrics()
        
    def _load_models_and_metrics(self):
        """Load all available models and their metrics"""
        logger.info("Loading models and metrics")
        
        # Find all categories with models
        categories = [d for d in os.listdir(self.models_dir) 
                     if os.path.isdir(os.path.join(self.models_dir, d))]
        
        for category in categories:
            try:
                # Load the model
                model_path = os.path.join(self.models_dir, category, 'model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        
                    # Load metrics
                    metrics_path = os.path.join(self.models_dir, category, 'metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                    else:
                        metrics = {}
                        
                    # Store in dictionaries
                    self.models[category] = model_data
                    self.metrics[category] = metrics
                    
                    logger.info(f"Loaded model for {category}")
            except Exception as e:
                logger.error(f"Error loading model for {category}: {str(e)}")
        
        logger.info(f"Loaded {len(self.models)} models")
        
    def load_category_benchmarks(self, file_path='logs/outlier_stats.json'):
        """
        Load category benchmarks from outlier stats
        
        This gives us important price distribution information
        for each category to inform pricing decisions
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    outlier_stats = json.load(f)
                    
                # Extract useful benchmark data
                for category, stats in outlier_stats.items():
                    # Get discounted_price stats
                    if 'discounted_price' in stats:
                        price_stats = stats['discounted_price']
                        
                        self.category_benchmarks[category] = {
                            'q1': float(price_stats['Q1']),
                            'q3': float(price_stats['Q3']),
                            'iqr': float(price_stats['IQR']),
                            'lower_bound': float(price_stats['lower_bound']),
                            'upper_bound': float(price_stats['upper_bound'])
                        }
                
                logger.info(f"Loaded price benchmarks for {len(self.category_benchmarks)} categories")
                return True
            else:
                logger.warning(f"Benchmark file not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading category benchmarks: {str(e)}")
            return False
    
    def predict_price(self, product_features, category):
        """
        Predict the market price for a product based on its features
        
        Parameters:
        -----------
        product_features : dict
            Dictionary of product features
        category : str
            Product category
            
        Returns:
        --------
        dict
            Dictionary with predicted price and confidence info
        """
        if category not in self.models:
            logger.error(f"No model available for category: {category}")
            return None
        
        try:
            # Get the model data
            model_data = self.models[category]
            model = model_data['model']
            scaler = model_data['scaler']
            is_log_price = model_data.get('is_log_price', False)
            
            # Convert dict to DataFrame
            features_df = pd.DataFrame([product_features])
            
            # Calculate derived features
            if 'manufacturing_cost' in features_df and 'price_to_cost_ratio' in features_df:
                # Estimated market price based on cost ratio
                features_df['category_median_price'] = features_df['manufacturing_cost'] * features_df['price_to_cost_ratio']
            
            if 'rating' in features_df and 'rating_count' in features_df:
                # Rating × count is a measure of review reliability
                features_df['rating_x_count'] = features_df['rating'] * features_df['rating_count']
                
                # Quality score derived from rating
                features_df['quality_tier'] = features_df['rating'] / 5.0  # Normalized 0-1
            
            if 'manufacturing_cost' in features_df:
                # Log transformed cost
                features_df['log_manufacturing_cost'] = np.log1p(features_df['manufacturing_cost'])
                
                # Use production_cost as synonym for manufacturing_cost
                features_df['production_cost'] = features_df['manufacturing_cost']
            
            if 'discount_percentage' in features_df and 'category_median_price' in features_df:
                # Estimated discounted price
                estimated_price = features_df['category_median_price']
                discount_pct = features_df['discount_percentage'] / 100.0
                features_df['discount_amount'] = estimated_price * discount_pct
                discounted_price = estimated_price * (1 - discount_pct)
                features_df['log_discounted_price'] = np.log1p(discounted_price)
            
            if 'brand_strength_score' in features_df:
                # Map brand_strength_score to brand_strength
                features_df['brand_strength'] = features_df['brand_strength_score']
            elif 'brand_strength' not in features_df:
                # Default brand strength - may be overridden by market conditions
                features_df['brand_strength'] = 0.5  # Medium
            
            # Add common derived features
            features_df['complexity_score'] = 0.5  # Medium complexity
            features_df['technology_level'] = 0.6  # Slightly above medium tech
            features_df['feature_count'] = 5      # Average feature count
            features_df['value_score'] = 0.5      # Medium value
            features_df['premium_index'] = 0.4    # Slightly below premium
            features_df['price_elasticity'] = -1.5  # Average elasticity
            features_df['estimated_units_sold'] = 1000  # Default units
            features_df['seasonal_relevance'] = 0.5  # Medium seasonality
            features_df['is_new_release'] = 0    # Not a new release
            
            # Calculate price ratios if possible
            if 'category_median_price' in features_df and 'manufacturing_cost' in features_df:
                if features_df['category_median_price'].iloc[0] > 0:
                    # Price relative to category median
                    estimated_price = features_df['manufacturing_cost'] * features_df['price_to_cost_ratio'] 
                    features_df['price_to_category_median_ratio'] = estimated_price / features_df['category_median_price']
                    features_df['price_relative_to_median'] = features_df['price_to_category_median_ratio'] - 1.0
                else:
                    features_df['price_to_category_median_ratio'] = 1.0
                    features_df['price_relative_to_median'] = 0.0
            
            # Calculate brand price ratio
            features_df['brand_avg_price'] = features_df['manufacturing_cost'] * 2.0  # Assume 2x markup
            features_df['brand_product_count'] = 100  # Default count
            features_df['price_to_brand_avg_ratio'] = 1.0  # Default ratio
            
            # Add market segmentation estimate
            price_segment_mapping = {
                'budget': 0,
                'value': 1,
                'mainstream': 2,
                'premium': 3,
                'luxury': 4
            }
            features_df['price_segment_numeric'] = 2  # Default to mainstream
            
            # Estimate price percentile position (0-1 scale)
            features_df['price_percentile'] = 0.5  # Default to median
            
            # Get the features that the model was trained on
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                
                # Create a DataFrame with all expected features
                model_features_df = pd.DataFrame(index=features_df.index)
                
                # Fill in the features from our calculations
                for feature in expected_features:
                    if feature in features_df.columns:
                        model_features_df[feature] = features_df[feature]
                    else:
                        logger.warning(f"Feature '{feature}' not found in input, setting to 0")
                        model_features_df[feature] = 0
                
                logger.info(f"Using {len(model_features_df.columns)} features for prediction")
                
                # Scale the features - ensure we respect column order
                try:
                    # Use column names when scaling to avoid the feature name warning
                    features_array = scaler.transform(model_features_df)
                except UserWarning:
                    # If there's still a warning, we'll silence it and continue
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, 
                                               message="X does not have valid feature names")
                        features_array = scaler.transform(model_features_df)
                
                # Make prediction
                predicted_value = model.predict(features_array)[0]
            else:
                # Fallback for models without feature_names_in_
                # Convert to numpy array and hope for the best
                features_array = scaler.transform(features_df)
                predicted_value = model.predict(features_array)[0]
            
            # If we used log transformation, convert back
            if is_log_price:
                predicted_price = np.expm1(predicted_value)
            else:
                predicted_price = predicted_value
            
            # Apply category-specific calibration factors
            # Based on observed price-to-median ratios from test results
            calibration_factors = {
                'Audio': 1.25,                # Observed 0.63x -> adjust by 1.25 to get closer to real prices
                'Cameras': 1.2,               # Observed 0.65x
                'Climate Control': 1.2,       # Observed 0.64x
                'Computers': 1.2,             # Observed 0.64x
                'Home Entertainment': 1.45,   # Observed 0.53x
                'Home Improvement': 1.2,      # Observed 0.63x
                'Home Office': 1.45,          # Observed 0.54x
                'Kitchen Appliances': 1.1,    # Observed 0.71x
                'Mobile Accessories': 1.2,    # Observed 0.69x
                'Smartwatches': 1.05          # Observed 0.77x (most accurate)
            }
            
            # Apply calibration if category exists in our factors
            if category in calibration_factors:
                calibration_factor = calibration_factors[category]
                predicted_price *= calibration_factor
                logger.info(f"Applied calibration factor of {calibration_factor:.2f} for {category}")
            
            # Sanity check on the predicted price
            if predicted_price <= 0:
                logger.warning(f"Invalid predicted price: {predicted_price}, using fallback estimation")
                # Fallback to a simple cost-plus model
                predicted_price = product_features['manufacturing_cost'] * product_features.get('price_to_cost_ratio', 2.0)
            
            # Get model accuracy metrics
            metrics = self.metrics.get(category, {})
            mape = metrics.get('test_mape', 0.15)  # Default to 15% if not available
            within_10pct = metrics.get('test_within_10pct', 0.8)  # Default to 80%
            
            # Calculate confidence range
            lower_bound = predicted_price * (1 - mape * 1.5)  # 1.5x MAPE for lower bound
            upper_bound = predicted_price * (1 + mape * 1.5)  # 1.5x MAPE for upper bound
            
            prediction_info = {
                'category': category,
                'predicted_market_price': float(predicted_price),
                'confidence_lower': float(lower_bound),
                'confidence_upper': float(upper_bound),
                'model_mape': float(mape),
                'model_within_10pct': float(within_10pct)
            }
            
            return prediction_info
            
        except Exception as e:
            logger.error(f"Error predicting price: {str(e)}")
            # Print detailed traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_competitive_price(self, prediction_info, manufacturing_cost, 
                             market_saturation='medium',
                             brand_strength='medium'):
        """
        Get a competitive price recommendation based on market prediction and strategy
        
        Parameters:
        -----------
        prediction_info : dict
            Output from predict_price method
        manufacturing_cost : float
            Manufacturing cost of the product
        market_saturation : str
            'low', 'medium', or 'high' - affects discount aggressiveness
        brand_strength : str
            'low', 'medium', or 'high' - affects discount aggressiveness
            
        Returns:
        --------
        dict
            Pricing recommendation with strategy details
        """
        if prediction_info is None:
            return None
        
        category = prediction_info['category']
        predicted_market_price = prediction_info['predicted_market_price']
        
        # Get thresholds for this category, defaulting to conservative values
        warning_threshold, viability_threshold = self.category_thresholds.get(
            category, (0.85, 1.0)  # Default thresholds if category not found
        )
        
        # Check if manufacturing cost is higher than market price based on category-specific threshold
        cost_to_market_ratio = manufacturing_cost / predicted_market_price
        is_cost_higher = cost_to_market_ratio > warning_threshold
        
        # Handle high manufacturing cost scenario
        if is_cost_higher:
            logger.warning(f"Manufacturing cost (₹{manufacturing_cost:.2f}) is {cost_to_market_ratio:.1f}x " + 
                          f"the predicted market price (₹{predicted_market_price:.2f})")
            
            # Get category-specific minimum margin or use default
            category_min_margin = self.category_min_margins.get(category, self.min_profit_margin)
            
            # Calculate minimum viable price with reduced profit margin for high-cost items
            # For high-cost scenarios, we can reduce the margin by up to 3%, but not below 3%
            min_profit_margin = max(0.03, category_min_margin - 0.03)
            logger.info(f"Using reduced minimum profit margin: {min_profit_margin:.1%} for high-cost item")
            
            # Calculate minimum viable price
            cost_plus_min_margin = manufacturing_cost / (1 - min_profit_margin)
            
            # Use category-specific viability threshold
            if cost_to_market_ratio > viability_threshold:
                logger.warning(f"Product may not be viable at this manufacturing cost")
                strategy_name = "Cost Reconsideration Required"
                
                # Calculate what manufacturing cost would need to be for viability
                viable_cost = predicted_market_price * (viability_threshold - 0.15)  # 15% below threshold
                
                recommendation = {
                    'category': category,
                    'predicted_market_price': float(predicted_market_price),
                    'manufacturing_cost': float(manufacturing_cost),
                    'recommended_price': float(cost_plus_min_margin),
                    'min_competitive_price': float(predicted_market_price * 0.9),  # 10% below market
                    'max_competitive_price': float(predicted_market_price),
                    'minimum_viable_price': float(cost_plus_min_margin),
                    'profit_margin': float((cost_plus_min_margin - manufacturing_cost) / cost_plus_min_margin),
                    'profit_margin_percentage': float(min_profit_margin * 100),
                    'discount_from_market': float((predicted_market_price - cost_plus_min_margin) / predicted_market_price * -100),
                    'market_saturation': market_saturation,
                    'brand_strength': brand_strength,
                    'price_elasticity': float(-1.2),  # Less elastic - premium pricing
                    'estimated_sales_impact': float(-15),  # Negative impact on sales volume
                    'strategy': strategy_name,
                    'viability_issue': True,
                    'recommended_max_cost': float(viable_cost),
                    'cost_reduction_needed': float(manufacturing_cost - viable_cost)
                }
                
                return recommendation
            
            # Cost is high but potentially viable with premium positioning
            strategy_name = "Premium Cost Recovery"
            recommended_price = cost_plus_min_margin * 1.05  # 5% above minimum viable
            
            recommendation = {
                'category': category,
                'predicted_market_price': float(predicted_market_price),
                'manufacturing_cost': float(manufacturing_cost),
                'recommended_price': float(recommended_price),
                'min_competitive_price': float(cost_plus_min_margin),
                'max_competitive_price': float(cost_plus_min_margin * 1.1),  # 10% above min viable
                'minimum_viable_price': float(cost_plus_min_margin),
                'profit_margin': float((recommended_price - manufacturing_cost) / recommended_price),
                'profit_margin_percentage': float(((recommended_price - manufacturing_cost) / recommended_price) * 100),
                'discount_from_market': float((predicted_market_price - recommended_price) / predicted_market_price * -100),
                'market_saturation': market_saturation,
                'brand_strength': brand_strength,
                'price_elasticity': float(-1.2),  # Less elastic for premium items
                'estimated_sales_impact': float(-15),  # Expect lower sales at higher price
                'strategy': strategy_name,
                'high_cost_warning': True
            }
            
            return recommendation
        
        # Normal competitive pricing scenario
        # Adjust discount factor based on market_saturation and brand_strength
        base_min, base_max = self.aggressive_discount_range
        
        # Market saturation adjustment - more saturated markets need higher discounts
        saturation_factor = {
            'low': -0.05,     # less discount needed
            'medium': 0,      # baseline
            'high': 0.05      # more discount needed
        }.get(market_saturation.lower(), 0)
        
        # Brand strength adjustment - stronger brands can use lower discounts
        brand_factor = {
            'low': 0.05,      # more discount needed
            'medium': 0,      # baseline
            'high': -0.05     # less discount needed
        }.get(brand_strength.lower(), 0)
        
        # Calculate adjusted discount range
        min_discount = base_min + saturation_factor + brand_factor
        max_discount = base_max + saturation_factor + brand_factor
        
        # Ensure discount range is reasonable
        min_discount = max(0.05, min(0.35, min_discount))  # Between 5-35%
        max_discount = max(0.10, min(0.40, max_discount))  # Between 10-40%
        
        # Calculate competitive price range
        competitive_min = predicted_market_price * (1 - max_discount)
        competitive_max = predicted_market_price * (1 - min_discount)
        
        # Get category-specific minimum margin or use default
        category_min_margin = self.category_min_margins.get(category, self.min_profit_margin)
        
        # Ensure minimum profit margin based on category
        cost_plus_min_margin = manufacturing_cost / (1 - category_min_margin)
        
        # Final recommended price shouldn't be lower than our minimum margin
        recommended_price = max(competitive_min, cost_plus_min_margin)
        
        # If recommended price is higher than max competitive price, adjust it
        if recommended_price > competitive_max:
            # Only if the difference is small, otherwise keep minimum viable price
            if (recommended_price - competitive_max) / competitive_max < 0.1:  # Less than 10% difference
                recommended_price = competitive_max
        
        # Calculate actual margin we'll get
        profit_margin = (recommended_price - manufacturing_cost) / recommended_price
        profit_margin_percentage = profit_margin * 100
        
        # Calculate discount from market price
        discount_from_market = ((predicted_market_price - recommended_price) / predicted_market_price) * 100
        
        # Calculate price elasticity (estimated)
        estimated_elasticity = -1.5  # Default elasticity
        if category in self.category_benchmarks:
            # Categories with wider price ranges tend to have higher elasticity
            q1 = self.category_benchmarks[category].get('q1', 0)
            q3 = self.category_benchmarks[category].get('q3', 0)
            if q1 > 0 and q3 > q1:
                price_range_ratio = q3 / q1
                # Scale elasticity based on price range (wider range = higher elasticity)
                estimated_elasticity = -1.0 - (0.5 * min(price_range_ratio / 3, 1.0))
        
        # Calculate expected sales impact
        sales_impact_percentage = -estimated_elasticity * discount_from_market
        
        # Get the percentile position in the market
        percentile_position = None
        if category in self.category_benchmarks:
            q1 = self.category_benchmarks[category].get('q1', 0)
            q3 = self.category_benchmarks[category].get('q3', 0)
            if q1 > 0 and q3 > q1:
                percentile_position = self._calculate_percentile_position(
                    recommended_price, q1, q3)
        
        # Get a descriptive name for the pricing strategy
        strategy_name = self._get_pricing_strategy_name(
            market_saturation, brand_strength, profit_margin, discount_from_market, category)
        
        # Build the recommendation object
        recommendation = {
            'category': category,
            'predicted_market_price': float(predicted_market_price),
            'manufacturing_cost': float(manufacturing_cost),
            'recommended_price': float(recommended_price),
            'min_competitive_price': float(competitive_min),
            'max_competitive_price': float(competitive_max),
            'minimum_viable_price': float(cost_plus_min_margin),
            'profit_margin': float(profit_margin),
            'profit_margin_percentage': float(profit_margin_percentage),
            'discount_from_market': float(discount_from_market),
            'market_saturation': market_saturation,
            'brand_strength': brand_strength,
            'price_elasticity': float(estimated_elasticity),
            'estimated_sales_impact': float(sales_impact_percentage),
            'strategy': strategy_name
        }
        
        # Add market position info if available
        if percentile_position is not None:
            recommendation['market_position_percentile'] = float(percentile_position)
        
        return recommendation
    
    def _calculate_percentile_position(self, price, q1, q3):
        """Calculate approximate percentile position based on quartile range"""
        if price <= q1:
            return 25 * (price / q1)
        elif price <= q3:
            return 25 + (50 * (price - q1) / (q3 - q1))
        else:
            return 75 + (25 * min((price - q3) / (q3 - q1), 1.0))
        
    def _get_pricing_strategy_name(self, market_saturation, brand_strength, 
                                  profit_margin, discount_percentage, category=None):
        """
        Get a descriptive name for the pricing strategy
        
        Parameters:
        -----------
        market_saturation : str
            'low', 'medium', or 'high'
        brand_strength : str
            'low', 'medium', or 'high'
        profit_margin : float
            Profit margin as a decimal
        discount_percentage : float
            Discount from market price as a percentage
        category : str, optional
            Product category for category-specific strategies
            
        Returns:
        --------
        str
            Strategy name
        """
        
        # Category-specific strategies
        if category:
            # Smartwatches can support higher margins
            if category == 'Smartwatches' and profit_margin > 0.35:
                return "Premium Brand Position"
            
            # Mobile accessories benefit from volume
            if category == 'Mobile Accessories' and discount_percentage > 20:
                return "High-Volume Entry Strategy"
                
            # Audio requires quality perception
            if category == 'Audio' and profit_margin > 0.3 and brand_strength == 'high':
                return "Premium Audio Experience"
                
            # Computers benefit from feature emphasis
            if category == 'Computers' and profit_margin < 0.15:
                return "Feature-Value Balance Strategy"
        
        # Base strategy on discount and profit margin
        if discount_percentage >= 25:
            base_strategy = "Aggressive Market Entry"
        elif discount_percentage >= 15:
            base_strategy = "Value-Oriented Strategy"
        elif discount_percentage >= 5:
            base_strategy = "Competitive Positioning"
        else:
            base_strategy = "Premium Positioning"
            
        # Consider market conditions
        if market_saturation == "high" and discount_percentage >= 20:
            return "Deep Discount Strategy"
        elif market_saturation == "high" and brand_strength == "low":
            return "Undercut Competitors Strategy"
        elif brand_strength == "high" and discount_percentage < 10:
            return "Brand Premium Strategy"
        elif profit_margin < 0.15:  # Less than 15% margin
            return "Thin-Margin Volume Strategy"
        else:
            return base_strategy
    
    def visualize_pricing_recommendation(self, recommendation, save_path=None):
        """
        Visualize the pricing recommendation with market context
        
        Parameters:
        -----------
        recommendation : dict
            The pricing recommendation
        save_path : str, optional
            If provided, the visualization will be saved to this path
            
        Returns:
        --------
        matplotlib.figure.Figure
            The visualization figure
        """
        try:
            category = recommendation['category']
            market_price = recommendation['predicted_market_price']
            recommended_price = recommendation['recommended_price']
            manufacturing_cost = recommendation['manufacturing_cost']
            min_competitive_price = recommendation['min_competitive_price']
            max_competitive_price = recommendation['max_competitive_price']
            minimum_viable_price = recommendation['minimum_viable_price']
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(12, 8))
            spec = fig.add_gridspec(2, 1, height_ratios=[2, 1])
            
            # Pricing strategy visualization (top)
            ax1 = fig.add_subplot(spec[0])
            
            # Plot price points
            price_points = [
                {'name': 'Manufacturing Cost', 'price': manufacturing_cost, 'color': 'gray'},
                {'name': 'Minimum Viable Price', 'price': minimum_viable_price, 'color': 'orange'},
                {'name': 'Min Competitive', 'price': min_competitive_price, 'color': 'green'},
                {'name': 'Recommended Price', 'price': recommended_price, 'color': 'blue'},
                {'name': 'Max Competitive', 'price': max_competitive_price, 'color': 'green'},
                {'name': 'Market Price', 'price': market_price, 'color': 'red'}
            ]
            
            # Sort by price
            price_points.sort(key=lambda x: x['price'])
            
            # Create y positions
            y_positions = np.linspace(0.2, 0.8, len(price_points))
            
            # Plot horizontal price line
            min_price = min(p['price'] for p in price_points) * 0.9
            max_price = max(p['price'] for p in price_points) * 1.1
            ax1.axhline(y=0.5, xmin=0, xmax=1, color='black', alpha=0.2, linestyle='--')
            
            # Plot price points
            for i, point in enumerate(price_points):
                y_pos = y_positions[i]
                ax1.scatter(point['price'], y_pos, color=point['color'], s=100, zorder=3)
                ax1.text(point['price'], y_pos+0.05, f"₹{point['price']:.2f}", 
                        ha='center', va='bottom', fontweight='bold')
                ax1.text(point['price'], y_pos-0.05, point['name'], 
                        ha='center', va='top')
            
            # Highlight the competitive range
            ax1.axvspan(min_competitive_price, max_competitive_price, alpha=0.2, color='green', zorder=1)
            ax1.text((min_competitive_price + max_competitive_price)/2, 0.1, 
                    'Competitive Range', ha='center', fontweight='bold')
            
            # Formatting
            ax1.set_xlim(min_price, max_price)
            ax1.set_ylim(0, 1)
            ax1.set_title(f'Pricing Strategy for {category}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Price (₹)', fontsize=12)
            ax1.xaxis.set_ticks_position('bottom')
            ax1.yaxis.set_visible(False)
            
            # Add profit margin and discount info
            profit_margin = recommendation['profit_margin_percentage']
            discount = recommendation['discount_from_market']
            strategy = recommendation['strategy']
            
            ax1.text(min_price + (max_price - min_price) * 0.02, 0.95, 
                    f"Strategy: {strategy}", fontsize=12, fontweight='bold')
            ax1.text(min_price + (max_price - min_price) * 0.02, 0.90, 
                    f"Profit Margin: {profit_margin:.1f}%", fontsize=12)
            ax1.text(min_price + (max_price - min_price) * 0.02, 0.85, 
                    f"Discount from Market: {discount:.1f}%", fontsize=12)
            
            # Price sensitivity curve (bottom)
            ax2 = fig.add_subplot(spec[1])
            
            # Get estimated elasticity
            elasticity = recommendation.get('price_elasticity', -1.5)
            
            # Create a range of discount percentages
            discount_range = np.linspace(0, 50, 100)
            
            # Calculate sales impact for each discount
            sales_impact = -elasticity * discount_range
            
            # Calculate net revenue impact (simplified model)
            # Revenue impact = (1 + sales_impact) * (1 - discount/100) - 1
            revenue_impact = (1 + sales_impact/100) * (1 - discount_range/100) - 1
            revenue_impact = revenue_impact * 100  # Convert to percentage
            
            # Plot the curves
            ax2.plot(discount_range, sales_impact, label='Sales Volume Impact', 
                    color='blue', linewidth=2)
            ax2.plot(discount_range, revenue_impact, label='Revenue Impact', 
                    color='green', linewidth=2)
            
            # Mark the recommended discount
            ax2.axvline(x=discount, color='red', linestyle='--', alpha=0.7)
            ax2.text(discount + 1, -5, f'{discount:.1f}% Discount', 
                    color='red', fontweight='bold')
            
            # Formatting
            ax2.set_xlim(0, 50)
            ax2.set_ylim(-40, 80)
            ax2.set_title('Price Sensitivity Analysis', fontsize=12)
            ax2.set_xlabel('Discount from Market Price (%)', fontsize=10)
            ax2.set_ylabel('Impact (%)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Add annotations for the estimated impact
            sales_impact_value = recommendation.get('estimated_sales_impact', 0)
            ax2.annotate(f'+{sales_impact_value:.1f}% Sales', 
                        xy=(discount, sales_impact_value), 
                        xytext=(discount+5, sales_impact_value+10),
                        arrowprops=dict(arrowstyle='->', color='blue'))
            
            # Annotate the revenue impact at the recommended discount
            revenue_impact_at_discount = (1 + sales_impact_value/100) * (1 - discount/100) - 1
            revenue_impact_at_discount = revenue_impact_at_discount * 100
            ax2.annotate(f'{revenue_impact_at_discount:.1f}% Revenue', 
                        xy=(discount, revenue_impact_at_discount), 
                        xytext=(discount+5, revenue_impact_at_discount-10),
                        arrowprops=dict(arrowstyle='->', color='green'))
            
            # Adjust layout - fix tight layout warning
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room at the top
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing pricing recommendation: {str(e)}")
            return None

def test_pricing_strategy():
    """Test function to demonstrate the pricing strategy"""
    try:
        logger.info("Testing pricing strategy implementation...")
        
        # Initialize strategy
        strategy = PricingStrategy()
        
        # Load benchmarks
        benchmark_loaded = strategy.load_category_benchmarks()
        logger.info(f"Benchmark loading {'successful' if benchmark_loaded else 'failed'}")
        
        # Test with a sample product
        sample_categories = list(strategy.models.keys())
        
        if not sample_categories:
            logger.error("No models available for testing")
            return
        
        logger.info(f"Available categories: {sample_categories}")
        sample_category = sample_categories[0]
        
        logger.info(f"Testing with sample category: {sample_category}")
        
        # Get model data to see what features it expects
        if sample_category in strategy.models:
            model_data = strategy.models[sample_category]
            if hasattr(model_data['model'], 'feature_names_in_'):
                logger.info(f"Model expects these features: {model_data['model'].feature_names_in_}")
        
        # Sample product features
        # Include more features to match what the model expects
        features = {
            'rating': 4.2,
            'rating_count': 120,
            'discount_percentage': 15,
            'manufacturing_cost': 100,
            'price_to_cost_ratio': 2.5,
            'margin_percentage': 40,
            'brand_strength_score': 0.6,
            'production_cost': 100,  # Same as manufacturing_cost
            'quality_score': 80      # Rating * 20
        }
        
        logger.info(f"Testing with features: {features}")
        
        # Test prediction
        prediction = strategy.predict_price(features, sample_category)
        
        if prediction:
            logger.info(f"Prediction successful: {prediction}")
            logger.info(f"Predicted market price: ₹{prediction['predicted_market_price']:.2f}")
            
            # Test recommendations with different scenarios
            scenarios = [
                {'name': 'New Brand, High Competition', 'saturation': 'high', 'strength': 'low'},
                {'name': 'Average Brand, Average Competition', 'saturation': 'medium', 'strength': 'medium'},
                {'name': 'Strong Brand, Low Competition', 'saturation': 'low', 'strength': 'high'}
            ]
            
            for scenario in scenarios:
                try:
                    logger.info(f"Testing scenario: {scenario['name']}")
                    recommendation = strategy.get_competitive_price(
                        prediction, 
                        features['manufacturing_cost'],
                        scenario['saturation'],
                        scenario['strength']
                    )
                    
                    if recommendation:
                        logger.info(f"Scenario: {scenario['name']}")
                        logger.info(f"Recommended price: ₹{recommendation['recommended_price']:.2f}")
                        logger.info(f"Discount from market: {recommendation['discount_from_market']:.1f}%")
                        logger.info(f"Profit margin: {recommendation['profit_margin_percentage']:.1f}%")
                        logger.info(f"Strategy: {recommendation['strategy']}")
                        
                        # Save visualization
                        try:
                            fig = strategy.visualize_pricing_recommendation(recommendation)
                            if fig:
                                save_path = os.path.join(
                                    'visualizations/pricing_strategies',
                                    f"{sample_category}_{scenario['saturation']}_{scenario['strength']}.png"
                                )
                                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                                plt.close(fig)
                                logger.info(f"Saved visualization to {save_path}")
                        except Exception as e:
                            logger.error(f"Error saving visualization: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in scenario {scenario['name']}: {str(e)}")
        else:
            logger.error("Failed to generate prediction")
            
    except Exception as e:
        logger.error(f"Error testing pricing strategy: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Test the implementation
    # First make sure directories exist
    os.makedirs('pricing_strategies', exist_ok=True)
    os.makedirs('visualizations/pricing_strategies', exist_ok=True)
    
    # Run the test
    test_pricing_strategy() 