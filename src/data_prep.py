import os
import pandas as pd
import numpy as np

def prepare_data(input_path, output_path):
    print(f"Starting data preparation: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
        
    # 1. Load dataset
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('%', 'pct')
    print("Cleaned column names to lowercase and underscores.")
    
    # 3. Handle missing values
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  - {col}: {count} missing values")
    
    # Fill numerical columns with median and categorical with mode as a fallback
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Imputed missing values in '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  Imputed missing values in '{col}' with mode: {mode_val}")
    
    # 4. Handle duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Found {duplicate_count} duplicate rows.")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Dropped duplicate rows.")
        
    # 5. Data Formatting & Feature Engineering
    # Parse date (format m/d/Y or similar)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract temporal features
    df['month'] = df['date'].dt.strftime('%B')
    df['month_num'] = df['date'].dt.month
    df['day_name'] = df['date'].dt.strftime('%A')
    df['day_of_week'] = df['date'].dt.dayofweek # Monday=0, Sunday=6
    df['year'] = df['date'].dt.year
    
    # Parse time (format H:M:S or I:M:S p)
    # Let's handle different time formats robustly
    try:
        # Time parsing
        df['time_parsed'] = pd.to_datetime(df['time'], format='%I:%M:%S %p', errors='coerce').dt.time
        # Fill coerce errors with 24h fallback
        null_times = df['time_parsed'].isnull()
        if null_times.any():
            df.loc[null_times, 'time_parsed'] = pd.to_datetime(df.loc[null_times, 'time'], format='%H:%M', errors='coerce').dt.time
        
        # Extract hour
        df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour
        # Fill missing hours
        df['hour'] = df['hour'].fillna(12).astype(int)
    except Exception as e:
        print(f"Warning parsing time: {e}. Extracting hour via string parsing.")
        # Fallback string split
        df['hour'] = df['time'].apply(lambda x: int(str(x).split(':')[0]) if ':' in str(x) else 12)
        
    # Define time of day category
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
            
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    
    # Validate mathematical columns (tax, sales, cogs)
    # sales = cogs + tax_5pct
    # gross_income = tax_5pct
    df['tax_5pct'] = df['tax_5pct'].round(4)
    df['cogs'] = df['cogs'].round(4)
    df['sales'] = df['sales'].round(4)
    df['gross_income'] = df['gross_income'].round(4)
    
    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved cleaned dataset to {output_path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    input_file = os.path.join("data", "SuperMarket Analysis.csv")
    output_file = os.path.join("dataset", "clean_supermarket_sales.csv")
    prepare_data(input_file, output_file)
