import os
import pandas as pd
import numpy as np
from scipy import stats

def run_statistical_analysis(input_path):
    print(f"Running statistical analysis on cleaned dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    results = {}
    
    # 1. Correlation Analysis (Pearson & Spearman)
    numeric_cols = ['unit_price', 'quantity', 'sales', 'cogs', 'gross_income', 'rating']
    
    pearson_corr = df[numeric_cols].corr(method='pearson')
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    results['pearson_corr'] = pearson_corr.to_dict()
    results['spearman_corr'] = spearman_corr.to_dict()
    
    print("\n--- Pearson Correlation Matrix ---")
    print(pearson_corr.round(3))
    
    # 2. Hypothesis Testing: T-Test (Gender vs. Sales)
    # H0: Mean sales of Female and Male are equal.
    # H1: Mean sales are different.
    female_sales = df[df['gender'] == 'Female']['sales']
    male_sales = df[df['gender'] == 'Male']['sales']
    
    t_stat_gender, p_val_gender = stats.ttest_ind(female_sales, male_sales, equal_var=False)
    results['t_test_gender'] = {
        't_stat': float(t_stat_gender),
        'p_value': float(p_val_gender),
        'female_mean': float(female_sales.mean()),
        'male_mean': float(male_sales.mean()),
        'reject_h0': bool(p_val_gender < 0.05)
    }
    
    print("\n--- Hypothesis Test: T-Test (Gender vs. Sales) ---")
    print(f"Female Mean Sales: ${female_sales.mean():.2f}")
    print(f"Male Mean Sales: ${male_sales.mean():.2f}")
    print(f"T-statistic: {t_stat_gender:.4f}, P-value: {p_val_gender:.4f}")
    print(f"Reject Null Hypothesis (at 5% level)? {'Yes' if p_val_gender < 0.05 else 'No'}")
    
    # 3. Hypothesis Testing: T-Test (Customer Type vs. Rating)
    # H0: Mean ratings of Members and Normal customers are equal.
    # H1: Mean ratings are different.
    member_rating = df[df['customer_type'] == 'Member']['rating']
    normal_rating = df[df['customer_type'] == 'Normal']['rating']
    
    t_stat_cust, p_val_cust = stats.ttest_ind(member_rating, normal_rating, equal_var=False)
    results['t_test_customer_rating'] = {
        't_stat': float(t_stat_cust),
        'p_value': float(p_val_cust),
        'member_mean': float(member_rating.mean()),
        'normal_mean': float(normal_rating.mean()),
        'reject_h0': bool(p_val_cust < 0.05)
    }
    
    print("\n--- Hypothesis Test: T-Test (Customer Type vs. Rating) ---")
    print(f"Member Mean Rating: {member_rating.mean():.2f}")
    print(f"Normal Mean Rating: {normal_rating.mean():.2f}")
    print(f"T-statistic: {t_stat_cust:.4f}, P-value: {p_val_cust:.4f}")
    print(f"Reject Null Hypothesis (at 5% level)? {'Yes' if p_val_cust < 0.05 else 'No'}")
    
    # 4. ANOVA: Sales across Branches
    # H0: Mean sales are identical across all branches.
    # H1: Mean sales are different for at least one branch.
    branches = df['branch'].unique()
    branch_groups = [df[df['branch'] == b]['sales'] for b in branches]
    
    f_stat_branch, p_val_branch = stats.f_oneway(*branch_groups)
    results['anova_branch'] = {
        'f_stat': float(f_stat_branch),
        'p_value': float(p_val_branch),
        'means': {b: float(df[df['branch'] == b]['sales'].mean()) for b in branches},
        'reject_h0': bool(p_val_branch < 0.05)
    }
    
    print("\n--- ANOVA: Sales across Branches ---")
    for b in branches:
        print(f"Branch {b} Mean Sales: ${df[df['branch'] == b]['sales'].mean():.2f}")
    print(f"F-statistic: {f_stat_branch:.4f}, P-value: {p_val_branch:.4f}")
    print(f"Reject Null Hypothesis (at 5% level)? {'Yes' if p_val_branch < 0.05 else 'No'}")
    
    # 5. ANOVA: Sales across Product Lines
    # H0: Mean sales are identical across all product lines.
    # H1: Mean sales are different for at least one product line.
    product_lines = df['product_line'].unique()
    prod_groups = [df[df['product_line'] == pl]['sales'] for pl in product_lines]
    
    f_stat_prod, p_val_prod = stats.f_oneway(*prod_groups)
    results['anova_product_line'] = {
        'f_stat': float(f_stat_prod),
        'p_value': float(p_val_prod),
        'means': {pl: float(df[df['product_line'] == pl]['sales'].mean()) for pl in product_lines},
        'reject_h0': bool(p_val_prod < 0.05)
    }
    
    print("\n--- ANOVA: Sales across Product Lines ---")
    for pl in product_lines:
        print(f"Product Line '{pl}' Mean Sales: ${df[df['product_line'] == pl]['sales'].mean():.2f}")
    print(f"F-statistic: {f_stat_prod:.4f}, P-value: {p_val_prod:.4f}")
    print(f"Reject Null Hypothesis (at 5% level)? {'Yes' if p_val_prod < 0.05 else 'No'}")
    
    return results

if __name__ == "__main__":
    input_file = os.path.join("dataset", "clean_supermarket_sales.csv")
    run_statistical_analysis(input_file)
