import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_production_recommendations(forecast_data, product_name="All Products", buffer_percentage=10):
    """
    Generate production recommendations based on the forecast data.
    
    Parameters:
    - forecast_data: Pandas DataFrame with forecast results from Prophet
    - product_name: Name of the product being forecasted
    - buffer_percentage: Percentage buffer to add to forecasts (default 10%)
    
    Returns:
    - recommendations: Dictionary containing production recommendations
    """
    # Create a copy of the forecast data
    forecast = forecast_data.copy()
    
    # Filter only future dates
    now = datetime.now()
    future_forecast = forecast[forecast['ds'] >= now].copy()
    
    # Calculate recommended production quantities with buffer
    future_forecast['recommended_production'] = np.ceil(future_forecast['yhat'] * (1 + buffer_percentage/100))
    
    # Calculate lower and upper bounds with buffer
    future_forecast['min_production'] = np.ceil(future_forecast['yhat_lower'] * (1 + buffer_percentage/100))
    future_forecast['max_production'] = np.ceil(future_forecast['yhat_upper'] * (1 + buffer_percentage/100))
    
    # Prepare daily production plan
    daily_plan = future_forecast[['ds', 'yhat', 'recommended_production', 'min_production', 'max_production']].copy()
    daily_plan.columns = ['Date', 'Forecasted Sales', 'Recommended Production', 'Minimum Production', 'Maximum Production']
    
    # Format date
    daily_plan['Date'] = daily_plan['Date'].dt.strftime('%Y-%m-%d')
    
    # Round values to integers
    daily_plan['Forecasted Sales'] = daily_plan['Forecasted Sales'].round().astype(int)
    daily_plan['Recommended Production'] = daily_plan['Recommended Production'].astype(int)
    daily_plan['Minimum Production'] = daily_plan['Minimum Production'].astype(int)
    daily_plan['Maximum Production'] = daily_plan['Maximum Production'].astype(int)
    
    # Identify risk days (days where the forecast confidence interval is wide)
    daily_plan['Uncertainty'] = daily_plan['Maximum Production'] - daily_plan['Minimum Production']
    daily_plan['Risk Level'] = 'Normal'
    
    # Calculate the average uncertainty
    avg_uncertainty = daily_plan['Uncertainty'].mean()
    std_uncertainty = daily_plan['Uncertainty'].std()
    
    # Mark high risk days (uncertainty > avg + 1.5*std)
    daily_plan.loc[daily_plan['Uncertainty'] > (avg_uncertainty + 1.5 * std_uncertainty), 'Risk Level'] = 'High'
    
    # Extract high risk days
    high_risk_days = daily_plan[daily_plan['Risk Level'] == 'High'].copy()
    
    # Generate risk assessment text
    if len(high_risk_days) > 0:
        risk_assessment = f"""
        **Risk Assessment**:
        - {len(high_risk_days)} days identified with high forecast uncertainty
        - These days may require special attention and flexible production planning
        - Consider having extra staff on hand or preparing more shelf-stable items
        """
    else:
        risk_assessment = "**Risk Assessment**: No high-risk days identified in the forecast period."
    
    # Calculate additional metrics
    avg_daily_production = daily_plan['Recommended Production'].mean()
    total_production = daily_plan['Recommended Production'].sum()
    peak_production = daily_plan['Recommended Production'].max()
    peak_production_date = daily_plan.loc[daily_plan['Recommended Production'].idxmax(), 'Date']
    
    # Generate additional recommendations
    additional_recommendations = generate_additional_recommendations(
        daily_plan, 
        product_name,
        avg_daily_production,
        peak_production
    )
    
    # Compile all recommendations into a dictionary
    recommendations = {
        'daily_plan': daily_plan[['Date', 'Forecasted Sales', 'Recommended Production', 'Risk Level']],
        'high_risk_days': high_risk_days[['Date', 'Forecasted Sales', 'Recommended Production', 'Uncertainty', 'Risk Level']],
        'avg_daily_production': avg_daily_production,
        'total_production': total_production,
        'peak_production': peak_production,
        'peak_production_date': peak_production_date,
        'buffer_percentage': buffer_percentage,
        'risk_assessment': risk_assessment,
        'additional_recommendations': additional_recommendations
    }
    
    return recommendations

def generate_additional_recommendations(daily_plan, product_name, avg_production, peak_production):
    """
    Generate additional context-specific recommendations based on the forecast.
    
    Parameters:
    - daily_plan: DataFrame with daily production recommendations
    - product_name: Name of the product being forecasted
    - avg_production: Average daily production
    - peak_production: Peak daily production
    
    Returns:
    - recommendations_text: String with additional recommendations
    """
    # Calculate day-to-day variability
    daily_plan['Prev_Production'] = daily_plan['Recommended Production'].shift(1)
    daily_plan['Daily_Change'] = daily_plan['Recommended Production'] - daily_plan['Prev_Production']
    daily_plan['Percent_Change'] = (daily_plan['Daily_Change'] / daily_plan['Prev_Production']) * 100
    
    # Drop NaN values (first row will have NaN for change calculation)
    daily_plan.dropna(subset=['Percent_Change'], inplace=True)
    
    # Find days with significant changes (more than 20%)
    significant_changes = daily_plan[abs(daily_plan['Percent_Change']) > 20]
    
    # Find the weekly pattern (which days have higher production)
    daily_plan['Day_Name'] = pd.to_datetime(daily_plan['Date']).dt.day_name()
    weekday_avg = daily_plan.groupby('Day_Name')['Recommended Production'].mean()
    highest_day = weekday_avg.idxmax()
    lowest_day = weekday_avg.idxmin()
    
    # Generate recommendations text
    recommendations = []
    
    # Product specific recommendation
    if product_name != "All Products":
        recommendations.append(f"**Product Specific ({product_name}):**")
    else:
        recommendations.append("**Overall Production:**")
    
    # Weekly pattern insights
    recommendations.append(f"- {highest_day}s show the highest average production requirements")
    recommendations.append(f"- {lowest_day}s show the lowest average production requirements")
    recommendations.append(f"- Consider adjusting staff scheduling to match this pattern")
    
    # Production variability insights
    if len(significant_changes) > 0:
        recommendations.append("\n**Production Variability:**")
        recommendations.append(f"- {len(significant_changes)} days show significant day-to-day production changes (>20%)")
        recommendations.append("- Consider preparing shelf-stable ingredients in advance for these fluctuations")
        recommendations.append("- Plan staff scheduling carefully around these dates")
    
    # Capacity planning insights
    recommendations.append("\n**Capacity Planning:**")
    recommendations.append(f"- Peak production day requires {peak_production} units")
    recommendations.append(f"- This is {((peak_production/avg_production)-1)*100:.1f}% higher than the average daily production")
    recommendations.append("- Ensure that equipment and staff capacity can handle peak days")
    recommendations.append("- Consider pre-producing stable components if peak exceeds production capacity")
    
    # Inventory management
    recommendations.append("\n**Inventory Management:**")
    recommendations.append("- Review ingredient inventory levels based on the forecast")
    recommendations.append("- Schedule deliveries to align with production peaks")
    recommendations.append("- Consider JIT (Just-In-Time) ordering for perishable ingredients")
    
    # Combine all recommendations
    recommendations_text = "\n".join(recommendations)
    
    return recommendations_text
