import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Import our pricing strategy implementation
sys.path.append('.')
from pricing_strategy import PricingStrategy

# Page configuration
st.set_page_config(
    page_title="Competitive Pricing Strategy Tool",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'pricing_strategy' not in st.session_state:
    st.session_state.pricing_strategy = PricingStrategy()
    # Load benchmarks
    st.session_state.pricing_strategy.load_category_benchmarks()
    
    # Debug info about benchmarks
    benchmark_info = {
        "Benchmark Categories": list(st.session_state.pricing_strategy.category_benchmarks.keys()),
        "Total Categories": len(st.session_state.pricing_strategy.category_benchmarks),
        "Sample Data": str(list(st.session_state.pricing_strategy.category_benchmarks.values())[:2])
    }
    st.sidebar.expander("Debug: Benchmark Data", expanded=False).write(benchmark_info)

# Sidebar
st.sidebar.image("logo.jpg", width=100)
st.sidebar.title("Pricing Strategy Tool")
st.sidebar.markdown("### For New Sellers")
st.sidebar.markdown("This tool helps new sellers enter the market with competitive pricing strategies.")

# Get available categories
available_categories = list(st.session_state.pricing_strategy.models.keys())

if not available_categories:
    st.error("No trained models available. Please run the model training scripts first.")
    st.stop()

# Main content
st.title("Competitive Pricing Strategy Tool")

# Add notification about model improvements
st.success("""
🚀 **Model Improvements**: We've enhanced the pricing engine with category-specific calibration, custom profit margins, 
and adaptive viability thresholds. These improvements make our recommendations more precise for each product category.
See the "Category Settings" tab for details.
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Product Pricing", "Batch Pricing", "Market Analysis", "Pricing Strategy History", "Category Settings"])

# Tab 1: Single Product Pricing
with tab1:
    st.header("Generate Pricing Recommendation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Basic product information
        category = st.selectbox("Product Category", available_categories)
        product_name = st.text_input("Product Name", "New Product")
        manufacturing_cost = st.number_input("Manufacturing Cost (₹)", min_value=1.0, value=100.0, step=10.0)
        
        # Display category-specific settings for the selected category
        if category in st.session_state.pricing_strategy.calibration_factors:
            with st.expander("Category-Specific Settings", expanded=False):
                settings_cols = st.columns(3)
                
                with settings_cols[0]:
                    calibration = st.session_state.pricing_strategy.calibration_factors.get(category, 1.0)
                    st.metric("Calibration Factor", f"{calibration:.2f}x")
                    st.caption("Adjusts price predictions for this category")
                
                with settings_cols[1]:
                    min_margin = st.session_state.pricing_strategy.category_min_margins.get(category, 0.1) * 100
                    st.metric("Min Profit Margin", f"{min_margin:.1f}%")
                    st.caption("Lowest acceptable profit margin")
                
                with settings_cols[2]:
                    warning, viability = st.session_state.pricing_strategy.category_thresholds.get(category, (0.85, 1.0))
                    st.metric("Warning Threshold", f"{warning:.2f}x")
                    st.metric("Viability Threshold", f"{viability:.2f}x")
                    st.caption("Cost-to-price ratio thresholds")
        
        # Market conditions
        st.subheader("Market Conditions")
        market_saturation = st.select_slider(
            "Market Saturation", 
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Low: Few competitors, High: Many competitors"
        )
        
        brand_strength = st.select_slider(
            "Brand Strength",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Low: New brand, High: Well-established brand"
        )
    
    with col2:
        # Product features
        st.subheader("Product Features")
        
        rating = st.slider("Product Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        rating_count = st.number_input("Number of Ratings", min_value=1, value=100, step=10)
        discount_percentage = st.slider("Current Market Discount %", min_value=0, max_value=70, value=15)
        
        # Advanced features (optional)
        with st.expander("Advanced Features"):
            price_to_cost_ratio = st.slider(
                "Industry Average Price-to-Cost Ratio", 
                min_value=1.1, 
                max_value=5.0, 
                value=2.5,
                step=0.1,
                help="Higher values mean higher markup is common in this category"
            )
            margin_percentage = st.slider(
                "Industry Average Margin %", 
                min_value=5, 
                max_value=80, 
                value=40,
                help="The average profit margin percentage in this category"
            )
    
    # Create features dictionary
    features = {
        'rating': rating,
        'rating_count': rating_count,
        'discount_percentage': discount_percentage,
        'manufacturing_cost': manufacturing_cost,
        'price_to_cost_ratio': price_to_cost_ratio,
        'margin_percentage': margin_percentage,
        'log_manufacturing_cost': np.log1p(manufacturing_cost)
    }
    
    # Button to generate pricing
    if st.button("Generate Pricing Strategy"):
        with st.spinner("Analyzing market conditions and generating recommendation..."):
            # Make prediction
            prediction = st.session_state.pricing_strategy.predict_price(features, category)
            
            if prediction:
                # Get recommendation
                recommendation = st.session_state.pricing_strategy.get_competitive_price(
                    prediction,
                    manufacturing_cost,
                    market_saturation.lower(),
                    brand_strength.lower()
                )
                
                if recommendation:
                    # Display results
                    st.session_state.recommendation = recommendation
                    
                    # Display the visualization
                    fig = st.session_state.pricing_strategy.visualize_pricing_recommendation(recommendation)
                    if fig:
                        st.pyplot(fig)
                        
                        # Save recommendation
                        filename = f"{product_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        filepath = os.path.join('pricing_strategies', f"{filename.replace(' ', '_')}.json")
                        os.makedirs('pricing_strategies', exist_ok=True)
                        
                        with open(filepath, 'w') as f:
                            json.dump(recommendation, f, indent=4)
                            
                        st.success(f"Pricing recommendation saved to {filepath}")
                        
                        # Display the actual recommendation values in a cleaner format
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Market Price", f"₹{recommendation['predicted_market_price']:.2f}")
                            st.metric("Manufacturing Cost", f"₹{recommendation['manufacturing_cost']:.2f}")
                        
                        with col2:
                            st.metric("Recommended Price", f"₹{recommendation['recommended_price']:.2f}")
                            st.metric("Discount from Market", f"{recommendation['discount_from_market']:.1f}%")
                        
                        with col3:
                            st.metric("Profit Margin", f"{recommendation['profit_margin_percentage']:.1f}%")
                            st.metric("Strategy", recommendation['strategy'])
                        
                        # Check for viability issues
                        if 'viability_issue' in recommendation and recommendation['viability_issue']:
                            st.error("""
                            ### ⚠️ Product Viability Issue
                            
                            The manufacturing cost is significantly higher than the market price, making it difficult to 
                            price this product competitively while maintaining profitability.
                            """)
                            
                            st.warning(f"""
                            **Recommended Action:**
                            - Consider reducing manufacturing cost to ₹{recommendation['recommended_max_cost']:.2f} or lower
                            - Current cost is ₹{recommendation['cost_reduction_needed']:.2f} too high for market viability
                            - Explore alternative product variations or premium positioning
                            """)
                        
                        # Check for high cost warning
                        elif 'high_cost_warning' in recommendation and recommendation['high_cost_warning']:
                            st.warning("""
                            ### ⚠️ High Manufacturing Cost
                            
                            The manufacturing cost is close to or above market price, requiring premium pricing strategy.
                            """)
                            
                            st.info(f"""
                            **Strategy Note:**
                            - Product will need to be positioned as premium quality to justify price
                            - Emphasize unique features and quality to overcome price sensitivity
                            - Consider cost optimization to improve margins in future production runs
                            """)
                        
                        # Regular recommendation display for normal cases
                        elif recommendation['discount_from_market'] < 0:
                            st.info(f"""
                            **Note:** This product is priced {abs(recommendation['discount_from_market']):.1f}% above the market price
                            due to manufacturing cost constraints. Consider ways to reduce production costs or position as premium.
                            """)
                            
                        # Add download button
                        st.download_button(
                            label="Download Recommendation as JSON",
                            data=json.dumps(recommendation, indent=4),
                            file_name=f"{filename}.json",
                            mime="application/json"
                        )
                else:
                    st.error("Failed to generate pricing recommendation.")
            else:
                st.error("Failed to predict market price.")

# Tab 2: Batch Pricing
with tab2:
    st.header("Batch Pricing")
    
    st.info("""
    📊 **Enhanced Batch Pricing**: Recommendations use category-specific calibration factors, minimum profit margins,
    and viability thresholds for more accurate results. Each product category has custom settings optimized for its market dynamics.
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file with products to price", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Check if required columns exist
            required_columns = ['category', 'product_name', 'manufacturing_cost']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Show expected format
                st.markdown("""
                ## Expected CSV Format:
                Your CSV should have the following columns:
                - `category`: The product category (must match available models)
                - `product_name`: Name of the product
                - `manufacturing_cost`: Manufacturing cost in dollars
                
                Optional columns:
                - `rating`: Product rating (1-5)
                - `rating_count`: Number of ratings
                - `discount_percentage`: Market discount percentage
                - `price_to_cost_ratio`: Industry average price to cost ratio
                - `margin_percentage`: Industry average margin percentage
                - `market_saturation`: "Low", "Medium", or "High"
                - `brand_strength`: "Low", "Medium", or "High"
                """)
                
                # Show sample format
                sample_data = {
                    'category': ['Computers', 'Audio', 'Mobile Accessories'],
                    'product_name': ['New Laptop', 'Wireless Earbuds', 'Phone Case'],
                    'manufacturing_cost': [450, 40, 5],
                    'rating': [4.2, 4.5, 4.0],
                    'rating_count': [120, 350, 980],
                    'discount_percentage': [12, 15, 25],
                    'market_saturation': ['Medium', 'High', 'High'],
                    'brand_strength': ['Low', 'Medium', 'Low']
                }
                
                sample_df = pd.DataFrame(sample_data)
                
                st.write("Sample format:")
                st.dataframe(sample_df)
                
                # Add download sample button
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="Download Sample CSV",
                    data=csv,
                    file_name="sample_batch_pricing.csv",
                    mime="text/csv"
                )
            else:
                # Fill default values for missing optional columns
                defaults = {
                    'rating': 4.0,
                    'rating_count': 100,
                    'discount_percentage': 15,
                    'price_to_cost_ratio': 2.5,
                    'margin_percentage': 40,
                    'market_saturation': 'Medium',
                    'brand_strength': 'Medium'
                }
                
                for col, default_val in defaults.items():
                    if col not in df.columns:
                        df[col] = default_val
                
                # Add log_manufacturing_cost
                df['log_manufacturing_cost'] = np.log1p(df['manufacturing_cost'])
                
                # Check for valid categories
                invalid_categories = [cat for cat in df['category'].unique() 
                                     if cat not in available_categories]
                
                if invalid_categories:
                    st.error(f"Invalid categories found: {', '.join(invalid_categories)}")
                    st.write(f"Available categories: {', '.join(available_categories)}")
                else:
                    # Generate batch recommendations button
                    if st.button("Generate Batch Recommendations"):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, row in df.iterrows():
                            try:
                                # Extract features
                                features = {
                                    'rating': row['rating'],
                                    'rating_count': row['rating_count'],
                                    'discount_percentage': row['discount_percentage'],
                                    'manufacturing_cost': row['manufacturing_cost'],
                                    'price_to_cost_ratio': row['price_to_cost_ratio'],
                                    'margin_percentage': row['margin_percentage'],
                                    'log_manufacturing_cost': row['log_manufacturing_cost']
                                }
                                
                                # Make prediction
                                prediction = st.session_state.pricing_strategy.predict_price(
                                    features, row['category'])
                                
                                if prediction:
                                    # Get recommendation
                                    recommendation = st.session_state.pricing_strategy.get_competitive_price(
                                        prediction,
                                        row['manufacturing_cost'],
                                        row['market_saturation'].lower(),
                                        row['brand_strength'].lower()
                                    )
                                    
                                    if recommendation:
                                        # Add product name
                                        recommendation['product_name'] = row['product_name']
                                        results.append(recommendation)
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(df))
                                
                            except Exception as e:
                                st.error(f"Error processing row {i}: {str(e)}")
                        
                        if results:
                            # Convert results to DataFrame for display
                            results_df = pd.DataFrame([
                                {
                                    'Product': r['product_name'],
                                    'Category': r['category'],
                                    'Market Price': f"₹{r['predicted_market_price']:.2f}",
                                    'Recommended Price': f"₹{r['recommended_price']:.2f}",
                                    'Discount': f"{r['discount_from_market']:.1f}%",
                                    'Profit Margin': f"{r['profit_margin_percentage']:.1f}%",
                                    'Strategy': r['strategy']
                                }
                                for r in results
                            ])
                            
                            st.success(f"Generated {len(results)} pricing recommendations")
                            st.dataframe(results_df)
                            
                            # Save all recommendations to a single file
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filepath = os.path.join('pricing_strategies', f"batch_pricing_{timestamp}.json")
                            
                            with open(filepath, 'w') as f:
                                json.dump(results, f, indent=4)
                            
                            # Create download link
                            st.download_button(
                                label="Download All Recommendations (JSON)",
                                data=json.dumps(results, indent=4),
                                file_name=f"batch_pricing_{timestamp}.json",
                                mime="application/json"
                            )
                            
                            # Create CSV download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv,
                                file_name=f"batch_pricing_{timestamp}.csv",
                                mime="text/csv"
                            )
                            
                            # Create visualizations for first 5 products
                            if len(results) > 0:
                                st.subheader("Sample Visualizations")
                                for i, recommendation in enumerate(results[:5]):
                                    st.write(f"### {recommendation['product_name']}")
                                    fig = st.session_state.pricing_strategy.visualize_pricing_recommendation(
                                        recommendation)
                                    if fig:
                                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 3: Market Analysis
with tab3:
    st.header("Market Analysis")
    
    # Select category for analysis
    analysis_category = st.selectbox(
        "Select Category for Analysis", 
        available_categories,
        key="analysis_category"
    )
    
    if st.button("Generate Market Analysis"):
        with st.spinner("Analyzing market data..."):
            try:
                # Use the category benchmarks already loaded in the pricing strategy
                market_stats = st.session_state.pricing_strategy.category_benchmarks
                
                if not market_stats:
                    st.error("No benchmark data loaded in the pricing strategy. Please check if category benchmarks were loaded correctly.")
                    st.stop()
                
                if analysis_category in market_stats:
                    # Create layout
                    col1, col2 = st.columns(2)
                    
                    # Extract data for selected category
                    cat_stats = market_stats[analysis_category]
                    
                    # Display summary statistics
                    with col1:
                        st.subheader("Price Distribution")
                        
                        # Create pretty metrics
                        metric_cols = st.columns(2)
                        with metric_cols[0]:
                            st.metric("Median Price", f"₹{cat_stats.get('median_price', 0):.2f}")
                            st.metric("25th Percentile (Q1)", f"₹{cat_stats.get('q1', 0):.2f}")
                            st.metric("75th Percentile (Q3)", f"₹{cat_stats.get('q3', 0):.2f}")
                        
                        with metric_cols[1]:
                            st.metric("Price Range", f"₹{cat_stats.get('max_price', 0) - cat_stats.get('min_price', 0):.2f}")
                            st.metric("Min Price", f"₹{cat_stats.get('min_price', 0):.2f}")
                            st.metric("Max Price", f"₹{cat_stats.get('max_price', 0):.2f}")
                        
                        # Show distribution visualization
                        try:
                            img_path = f"visualizations/{analysis_category}_price_distribution.png"
                            if os.path.exists(img_path):
                                st.image(img_path, use_column_width=True)
                            else:
                                st.info("Distribution visualization not available")
                        except Exception as e:
                            st.error(f"Error displaying distribution: {str(e)}")
                    
                    # Create price elasticity analysis
                    with col2:
                        st.subheader("Price Elasticity Analysis")
                        
                        # Calculate estimated elasticity
                        q1 = cat_stats.get('q1', 0)
                        q3 = cat_stats.get('q3', 0)
                        
                        if q1 > 0 and q3 > q1:
                            price_range_ratio = q3 / q1
                            estimated_elasticity = -1.0 - (0.5 * min(price_range_ratio / 3, 1.0))
                        else:
                            estimated_elasticity = -1.5
                        
                        st.metric("Estimated Price Elasticity", f"{estimated_elasticity:.2f}")
                        st.info("""
                        **Price Elasticity Interpretation:**
                        * Values closer to -1.0 indicate less elastic (less sensitive to price)
                        * Values below -1.5 indicate highly elastic (very sensitive to price)
                        """)
                        
                        # Create elasticity visualization
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        # Create a range of discount percentages
                        discount_range = np.linspace(0, 50, 100)
                        
                        # Calculate sales impact for each discount
                        sales_impact = -estimated_elasticity * discount_range
                        
                        # Calculate net revenue impact
                        revenue_impact = (1 + sales_impact/100) * (1 - discount_range/100) - 1
                        revenue_impact = revenue_impact * 100  # Convert to percentage
                        
                        # Plot the curves
                        ax.plot(discount_range, sales_impact, label='Sales Volume Impact', 
                                color='blue', linewidth=2)
                        ax.plot(discount_range, revenue_impact, label='Revenue Impact', 
                                color='green', linewidth=2)
                        
                        # Find the optimal discount (maximum revenue)
                        optimal_discount_idx = np.argmax(revenue_impact)
                        optimal_discount = discount_range[optimal_discount_idx]
                        max_revenue_impact = revenue_impact[optimal_discount_idx]
                        
                        # Mark the optimal discount
                        ax.axvline(x=optimal_discount, color='red', linestyle='--', alpha=0.7)
                        ax.scatter(optimal_discount, max_revenue_impact, color='red', s=100, zorder=5)
                        ax.annotate(f'Optimal: {optimal_discount:.1f}%',
                                   xy=(optimal_discount, max_revenue_impact),
                                   xytext=(optimal_discount+5, max_revenue_impact+5),
                                   arrowprops=dict(arrowstyle='->', color='red'))
                        
                        # Formatting
                        ax.set_xlim(0, 50)
                        ax.set_ylim(-40, 80)
                        ax.set_title('Price Sensitivity Analysis', fontsize=12)
                        ax.set_xlabel('Discount from Market Price (%)', fontsize=10)
                        ax.set_ylabel('Impact (%)', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='upper right')
                        
                        st.pyplot(fig)
                        
                        # Add interpretation
                        st.markdown(f"""
                        **Market Analysis Insights:**
                        
                        * **Optimal Discount:** {optimal_discount:.1f}% below market price maximizes revenue
                        * **Expected Sales Increase:** {-estimated_elasticity * optimal_discount:.1f}% at optimal price point
                        * **Expected Revenue Impact:** {max_revenue_impact:.1f}% at optimal price point
                        """)
                        
                    # Add competitive strategy recommendations
                    st.subheader("Competitive Strategy Recommendations")
                    
                    # Determine market characteristics
                    price_range_width = (cat_stats.get('max_price', 0) - cat_stats.get('min_price', 0)) / cat_stats.get('median_price', 1)
                    
                    # Different strategies based on price range and elasticity
                    strategies = []
                    
                    if price_range_width > 2.0:
                        strategies.append({
                            "name": "Market Segmentation Strategy",
                            "description": "Wide price range indicates multiple market segments. Consider different product variants targeting specific price points.",
                            "discount_range": "Varies by segment",
                            "ideal_for": "Products with customizable features or tiered offerings"
                        })
                    
                    if estimated_elasticity < -1.7:
                        strategies.append({
                            "name": "Aggressive Value Leader",
                            "description": "High price sensitivity indicates aggressive discounting can capture significant market share.",
                            "discount_range": "20-30% below market median",
                            "ideal_for": "New market entrants with efficient cost structure"
                        })
                    elif estimated_elasticity > -1.3:
                        strategies.append({
                            "name": "Quality Differentiator",
                            "description": "Lower price sensitivity suggests focusing on quality and features rather than price.",
                            "discount_range": "0-10% below market median",
                            "ideal_for": "Products with strong unique features or brand reputation"
                        })
                    else:
                        strategies.append({
                            "name": "Balanced Value Approach",
                            "description": "Moderate price sensitivity allows for competitive pricing without extreme discounting.",
                            "discount_range": "10-20% below market median",
                            "ideal_for": "Products with good balance of features and price"
                        })
                    
                    # Add specific recommendation for market entrants
                    strategies.append({
                        "name": "New Seller Strategy",
                        "description": "For new sellers entering this category, build market presence with an initial discount that gradually reduces as reputation grows.",
                        "discount_range": f"{optimal_discount:.1f}% initially, reducing by 5% every 3 months",
                        "ideal_for": "New sellers with limited reviews and brand recognition"
                    })
                    
                    # Display strategies
                    for i, strategy in enumerate(strategies):
                        with st.expander(f"{i+1}. {strategy['name']}", expanded=i==0):
                            st.markdown(f"**Description:** {strategy['description']}")
                            st.markdown(f"**Recommended Discount Range:** {strategy['discount_range']}")
                            st.markdown(f"**Ideal For:** {strategy['ideal_for']}")
                else:
                    st.error(f"No market data available for {analysis_category}")
                    
                    # Show available categories
                    available_categories_in_stats = list(market_stats.keys())
                    if available_categories_in_stats:
                        st.info(f"Data is available for these categories: {', '.join(available_categories_in_stats)}")
                    
                    # Show debugging info
                    st.expander("Debugging Information", expanded=False).write({
                        "Available Stats Categories": list(market_stats.keys()),
                        "Selected Category": analysis_category,
                        "Stats Data Sample": str(market_stats)[:1000] + "..." if len(str(market_stats)) > 1000 else str(market_stats)
                    })
            except Exception as e:
                st.error(f"Error generating market analysis: {str(e)}")
                st.error("Please check that category benchmarks have been loaded properly.")

# Add a batch upload page for pricing strategies
with tab4:
    st.header("Pricing Strategy History")
    
    # Show previously saved pricing strategies
    try:
        pricing_files = sorted(
            [f for f in os.listdir('pricing_strategies') if f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join('pricing_strategies', x)),
            reverse=True
        )
        
        if pricing_files:
            st.write(f"Found {len(pricing_files)} saved pricing strategies")
            
            # Create a filter by category
            all_saved_categories = set()
            
            # Read the first few strategies to extract categories
            for file in pricing_files[:min(len(pricing_files), 20)]:
                try:
                    with open(os.path.join('pricing_strategies', file), 'r') as f:
                        strategy_data = json.load(f)
                        all_saved_categories.add(strategy_data.get('category', 'Unknown'))
                except:
                    pass
            
            # Filter options
            filter_category = st.selectbox(
                "Filter by Category",
                options=["All Categories"] + sorted(list(all_saved_categories))
            )
            
            # Display strategies
            for file in pricing_files:
                try:
                    file_path = os.path.join('pricing_strategies', file)
                    with open(file_path, 'r') as f:
                        strategy_data = json.load(f)
                    
                    strategy_category = strategy_data.get('category', 'Unknown')
                    
                    # Apply category filter
                    if filter_category != "All Categories" and strategy_category != filter_category:
                        continue
                    
                    # Format creation time
                    creation_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Create an expander for each strategy
                    with st.expander(f"{strategy_data.get('category', 'Unknown')} - {file.split('_')[0]}"):
                        # Display in columns
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write(f"**Created:** {creation_time.strftime('%Y-%m-%d %H:%M')}")
                            st.metric("Market Price", f"₹{strategy_data.get('predicted_market_price', 0):.2f}")
                            st.metric("Recommended Price", f"₹{strategy_data.get('recommended_price', 0):.2f}")
                            st.metric("Discount", f"{strategy_data.get('discount_from_market', 0):.1f}%")
                            st.metric("Profit Margin", f"{strategy_data.get('profit_margin_percentage', 0):.1f}%")
                            st.write(f"**Strategy:** {strategy_data.get('strategy', 'Unknown')}")
                        
                        with col2:
                            # Re-generate visualization
                            if strategy_data:
                                pricing_strategy = st.session_state.pricing_strategy
                                fig = pricing_strategy.visualize_pricing_recommendation(strategy_data)
                                if fig:
                                    st.pyplot(fig)
                        
                        # Provide download button for the strategy
                        st.download_button(
                            "Download Strategy JSON",
                            data=json.dumps(strategy_data, indent=4),
                            file_name=file,
                            mime="application/json"
                        )
                        
                except Exception as e:
                    st.error(f"Error loading strategy {file}: {str(e)}")
        else:
            st.info("No saved pricing strategies found. Generate some pricing recommendations first!")
    except Exception as e:
        st.error(f"Error reading pricing strategies: {str(e)}")

# Add new tab for Category-Specific settings
with tab5:
    st.header("Category-Specific Settings")
    
    # Get category-specific settings from the pricing strategy object
    if 'pricing_strategy' in st.session_state:
        pricing_strategy = st.session_state.pricing_strategy
        
        # Display calibration factors
        st.subheader("Calibration Factors")
        calibration_data = []
        for category, factor in pricing_strategy.calibration_factors.items():
            calibration_data.append({"Category": category, "Calibration Factor": f"{factor:.2f}x"})
        
        calibration_df = pd.DataFrame(calibration_data)
        st.dataframe(calibration_df)
        
        # Display minimum profit margins
        st.subheader("Minimum Profit Margins")
        margin_data = []
        for category, margin in pricing_strategy.category_min_margins.items():
            margin_data.append({"Category": category, "Min Profit Margin": f"{margin*100:.1f}%"})
        
        margin_df = pd.DataFrame(margin_data)
        st.dataframe(margin_df)
        
        # Display viability thresholds
        st.subheader("Viability Thresholds")
        threshold_data = []
        for category, thresholds in pricing_strategy.category_thresholds.items():
            warning, viability = thresholds
            threshold_data.append({
                "Category": category, 
                "Warning Threshold": f"{warning:.2f}x", 
                "Viability Threshold": f"{viability:.2f}x"
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        st.dataframe(threshold_df)
        
        # Add visualization for minimum profit margins
        st.subheader("Minimum Profit Margin Comparison")
        
        # Prepare data for visualization
        margin_compare_data = []
        for category, margin in pricing_strategy.category_min_margins.items():
            margin_compare_data.append({"Category": category, "Minimum Profit Margin (%)": margin*100})
        
        margin_compare_df = pd.DataFrame(margin_compare_data)
        
        # Sort by margin for better visualization
        margin_compare_df = margin_compare_df.sort_values("Minimum Profit Margin (%)")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(margin_compare_df)))
        
        bars = ax.barh(margin_compare_df["Category"], margin_compare_df["Minimum Profit Margin (%)"], color=colors)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    ha='left', va='center', fontweight='bold')
        
        # Formatting
        ax.set_title("Minimum Profit Margins by Category", fontsize=14)
        ax.set_xlabel("Minimum Profit Margin (%)", fontsize=12)
        ax.set_ylabel("Product Category", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
        # Add visualization for calibration factors
        st.subheader("Calibration Factor Comparison")
        
        # Prepare data for visualization
        calib_compare_data = []
        for category, factor in pricing_strategy.calibration_factors.items():
            calib_compare_data.append({"Category": category, "Calibration Factor": factor})
        
        calib_compare_df = pd.DataFrame(calib_compare_data)
        
        # Sort by factor for better visualization
        calib_compare_df = calib_compare_df.sort_values("Calibration Factor")
        
        # Create bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors2 = plt.cm.cool(np.linspace(0.2, 0.8, len(calib_compare_df)))
        
        bars2 = ax2.barh(calib_compare_df["Category"], calib_compare_df["Calibration Factor"], color=colors2)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.2f}x", 
                     ha='left', va='center', fontweight='bold')
        
        # Formatting
        ax2.set_title("Price Calibration Factors by Category", fontsize=14)
        ax2.set_xlabel("Calibration Factor", fontsize=12)
        ax2.set_ylabel("Product Category", fontsize=12)
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        st.pyplot(fig2)
        
        # Add description of settings
        st.markdown("""
        ### About These Settings
        
        **Calibration Factors**: These factors adjust the predicted market price to ensure accurate recommendations for each category. 
        Higher values mean a larger adjustment to the base model prediction.
        
        **Minimum Profit Margins**: The lowest acceptable profit margin for each product category. 
        Categories with higher brand value or quality perception have higher minimum margins.
        
        **Viability Thresholds**:
        * **Warning Threshold**: When manufacturing cost exceeds this percentage of market price, the system issues a warning
        * **Viability Threshold**: When manufacturing cost exceeds this percentage of market price, the product is flagged as potentially not viable
        
        These settings are calibrated based on category-specific market analysis and profitability requirements.
        """)
    else:
        st.info("Pricing strategy not initialized")

# Footer
st.markdown("---")
st.markdown("Competitive Pricing Strategy Tool - Built with Streamlit")
st.markdown("Version 2.0 - Enhanced with category-specific calibration, margins and thresholds") 