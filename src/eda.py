import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set beautiful styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

# Custom premium palette
BRAND_PALETTE = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5"]
sns.set_palette(sns.color_palette(BRAND_PALETTE))

def run_eda(input_path, output_dir):
    print(f"Running EDA on cleaned dataset: {input_path}")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Total Sales Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='sales', kde=True, color='#1f77b4', bins=30)
    plt.title('Distribution of Sales (Invoice Value)', pad=15)
    plt.xlabel('Sales (USD)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Branch Performance (Revenue and Gross Income)
    branch_perf = df.groupby('branch')[['sales', 'gross_income']].sum().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(ax=axes[0], data=branch_perf, x='branch', y='sales', hue='branch', palette='viridis', legend=False)
    axes[0].set_title('Total Revenue by Branch', pad=10)
    axes[0].set_ylabel('Total Revenue (USD)')
    axes[0].set_xlabel('Branch')
    
    sns.barplot(ax=axes[1], data=branch_perf, x='branch', y='gross_income', hue='branch', palette='magma', legend=False)
    axes[1].set_title('Total Gross Income (Profit) by Branch', pad=10)
    axes[1].set_ylabel('Total Profit (USD)')
    axes[1].set_xlabel('Branch')
    plt.suptitle('Branch Performance Comparison', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'branch_performance.png'), dpi=150)
    plt.close()
    
    # 3. Product Line Sales & Quantity
    prod_perf = df.groupby('product_line')[['sales', 'quantity']].sum().sort_values(by='sales', ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=prod_perf, y='product_line', x='sales', palette='plasma')
    plt.title('Total Revenue by Product Line', pad=15)
    plt.xlabel('Total Revenue (USD)')
    plt.ylabel('Product Line')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'product_line_sales.png'), dpi=150)
    plt.close()
    
    # 4. Customer Type & Gender Sales Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='customer_type', y='sales', hue='gender', palette='Set2')
    plt.title('Sales Distribution by Customer Type and Gender', pad=15)
    plt.xlabel('Customer Type')
    plt.ylabel('Sales (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'customer_gender_sales.png'), dpi=150)
    plt.close()
    
    # 5. Monthly Revenue Trend
    monthly_sales = df.groupby('month')[['sales', 'gross_income']].sum().reindex(['January', 'February', 'March']).reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_sales, x='month', y='sales', marker='o', linewidth=2.5, color='#d62728', label='Revenue')
    plt.title('Monthly Sales Trend (Q1 2019)', pad=15)
    plt.xlabel('Month')
    plt.ylabel('Total Sales (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_sales_trend.png'), dpi=150)
    plt.close()
    
    # 6. Payment Method Distribution
    payment_counts = df['payment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=['#1f77b4', '#aec7e8', '#ff7f0e'], textprops={'fontsize': 12},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Payment Method Distribution', pad=15, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'payment_distribution.png'), dpi=150)
    plt.close()
    
    # 7. Customer Ratings by Product Line
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='rating', y='product_line', palette='coolwarm')
    plt.title('Customer Rating Distribution by Product Line', pad=15)
    plt.xlabel('Rating (0-10)')
    plt.ylabel('Product Line')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_product_line.png'), dpi=150)
    plt.close()
    
    # 8. Sales by Time of Day
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='time_of_day', order=time_order, palette='muted')
    plt.title('Transaction Count by Time of Day', pad=15)
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transactions_by_time_of_day.png'), dpi=150)
    plt.close()
    
    # 9. Correlation Heatmap
    # Filter numeric cols
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Drop month_num and day_of_week and year for cleaner heat map
    cols_to_drop = ['month_num', 'day_of_week', 'year']
    numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, square=True)
    plt.title('Correlation Matrix of Numerical Features', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
    plt.close()
    
    # 10. Average Rating by Branch
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='branch', y='rating', hue='branch', palette='Set1', errorbar=None, legend=False)
    plt.title('Average Customer Rating by Branch', pad=15)
    plt.xlabel('Branch')
    plt.ylabel('Average Rating')
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'branch_ratings.png'), dpi=150)
    plt.close()

    # 11. Quantity / Basket Size Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='quantity', color='#2ca02c')
    plt.title('Distribution of Quantity per Invoice (Basket Size)', pad=15)
    plt.xlabel('Quantity Purchased')
    plt.ylabel('Transaction Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'basket_size_distribution.png'), dpi=150)
    plt.close()
    
    # 12. City Performance Comparison (Revenue vs Ratings)
    city_perf = df.groupby('city').agg({'sales': 'sum', 'rating': 'mean'}).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(ax=axes[0], data=city_perf, x='city', y='sales', hue='city', palette='crest', legend=False)
    axes[0].set_title('Total Revenue by City', pad=10)
    axes[0].set_ylabel('Total Revenue (USD)')
    
    sns.barplot(ax=axes[1], data=city_perf, x='city', y='rating', hue='city', palette='flare', legend=False)
    axes[1].set_title('Average Customer Rating by City', pad=10)
    axes[1].set_ylabel('Average Rating')
    axes[1].set_ylim(0, 10)
    
    plt.suptitle('City Sales and Customer Rating Performance', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'city_performance.png'), dpi=150)
    plt.close()
    
    print("All 12 EDA visualizations generated and saved successfully!")

if __name__ == "__main__":
    input_file = os.path.join("dataset", "clean_supermarket_sales.csv")
    output_directory = "images"
    run_eda(input_file, output_directory)
