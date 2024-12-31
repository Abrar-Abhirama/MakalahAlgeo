import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import time

def load_and_preprocess_data(file_path):
    print("Reading data...")
    # Read data
    df = pd.read_csv(file_path, sep=';', 
                     dtype={ 
                         'Global_active_power': str,
                         'Global_reactive_power': str,
                         'Voltage': str,
                         'Global_intensity': str,
                         'Sub_metering_1': str,
                         'Sub_metering_2': str,
                         'Sub_metering_3': str
                     })
    
    print("Processing datetime...")
    # Convert datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                    format='%d/%m/%Y %H:%M:%S', 
                                    dayfirst=True)
    print("Converting numeric columns...")
    # Convert numeric columns
    numeric_columns = ['Global_active_power', 
                       'Global_reactive_power', 'Voltage', 
                      'Global_intensity', 'Sub_metering_1', 
                      'Sub_metering_2', 'Sub_metering_3']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', '.'), 
                                errors='coerce')
    print("Adding hour column...")
    # Add hour column
    df['Hour'] = df['DateTime'].dt.hour  # Extract hour only
    print("Handling missing values...")
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')  
    return df

def create_hourly_consumption_matrix(df):
    print("Creating consumption matrix...")
    # Group by date and hour, calculate mean consumption
    hourly_data = df.groupby([df['DateTime'].dt.date, df['Hour']])['Global_active_power'].mean()
    print(f"Hourly data shape: {hourly_data.shape}")

    # Pivot to create matrix (days x hours)
    consumption_matrix = hourly_data.unstack()
    print(f"Consumption matrix shape: {consumption_matrix.shape}")
    print(consumption_matrix.head())

    # Fill any remaining NaN values with column means
    consumption_matrix = consumption_matrix.fillna(consumption_matrix.mean())
    print("Consumption matrix after filling NaN values:")
    print(consumption_matrix.head())
    
    return consumption_matrix

def analyze_daily_patterns(df):
    """
    Analyze daily consumption patterns by hour
    """
    print("Analyzing daily patterns...")
    # Calculate average consumption by hour
    hourly_avg = df.groupby('Hour')['Global_active_power'].mean()
    
    # Calculate peak, minimum, and maximum consumption
    max_consumption = hourly_avg.max()
    min_consumption = hourly_avg.min()
    
    # Calculate threshold for peak detection
    threshold = hourly_avg.mean() + hourly_avg.std()
    peak_hours = hourly_avg[hourly_avg > threshold].index.tolist()
    
    return hourly_avg, peak_hours, threshold, max_consumption, min_consumption

def identify_peak_periods(df, threshold):
    print("Identifying peak periods...")
    print(f"Threshold for peak periods: {threshold}")
    peak_periods = df[df['Global_active_power'] > threshold]
    print(f"Number of peak periods identified: {len(peak_periods)}")
    return peak_periods[['DateTime', 'Global_active_power', 'Hour']]

def plot_results(hourly_avg, peak_hours, threshold, max_consumption, min_consumption, pca, explained_variance_ratio):
    """
    Create visualizations for daily patterns and PCA
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average Daily Pattern
    plt.subplot(2, 2, 1)
    plt.plot(hourly_avg.index, hourly_avg.values, label='Hourly Avg Consumption', marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Peak Threshold')
    plt.axhline(y=max_consumption, color='g', linestyle='--', label='Maximum Consumption')
    plt.axhline(y=min_consumption, color='b', linestyle='--', label='Minimum Consumption')
    plt.title('Average Daily Consumption Pattern')
    plt.xlabel('Hour of Day')
    plt.xticks(range(0, 24))  # Ensure only 24 hours are shown on the x-axis
    plt.ylabel('Average Global Active Power (kW)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: PCA Explained Variance
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
             np.cumsum(explained_variance_ratio), 
             'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    # Plot 3: First Principal Component
    plt.subplot(2, 2, 3)
    plt.plot(range(24), pca.components_[0], 'g-')
    plt.title('First Principal Component Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel('Component Weight')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def perform_pca_analysis(consumption_matrix):
    """
    Perform PCA analysis on hourly consumption patterns
    """
    print("Performing PCA...")
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(consumption_matrix)
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    return pca, pca.explained_variance_ratio_, data_scaled

def main(file_path):
    try:
        start_time = time.time()
        
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        print(f"Data loading time: {time.time() - start_time:.2f} seconds")
        
        # Create consumption matrix
        consumption_matrix = create_hourly_consumption_matrix(df)
        
        # Perform PCA
        pca, explained_variance_ratio, data_scaled = perform_pca_analysis(consumption_matrix)
        
        # Analyze daily patterns
        hourly_avg, peak_hours, threshold, max_consumption, min_consumption = analyze_daily_patterns(df)
        
        # Identify peak periods
        peak_periods = identify_peak_periods(df, threshold)
        
        # Plot results
        plot_results(hourly_avg, peak_hours, threshold, max_consumption, min_consumption, pca, explained_variance_ratio)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Peak consumption hours: {sorted(peak_hours)}")
        print(f"Average consumption: {hourly_avg.mean():.2f} kW")
        print(f"Maximum consumption: {max_consumption:.2f} kW (Hour: {hourly_avg.idxmax()})")
        print(f"Minimum consumption: {min_consumption:.2f} kW (Hour: {hourly_avg.idxmin()})")
        print(f"Peak periods (above threshold):\n{peak_periods.head(10)}")  # Print top 10 peak periods
        
        print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = "household_power_consumption.txt"  # Replace with your file path
    main(file_path)
