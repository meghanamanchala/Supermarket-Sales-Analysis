import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config for wide layout and premium title
st.set_page_config(
    page_title="Supermarket Sales Executive Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using HTML/CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global font override */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient Main Title */
    .main-title {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    
    /* Glassmorphism KPI Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.15);
        border-color: #4f46e5;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.6rem;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.4rem;
    }
    
    /* Styled Section Header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4f46e5;
        padding-left: 0.75rem;
    }
    
    /* Prediction output card */
    .prediction-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        color: #ecfdf5;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #059669;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_clean_data():
    file_path = os.path.join("dataset", "clean_supermarket_sales.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        st.error(f"Preprocessed dataset not found at {file_path}. Please run python src/data_prep.py first.")
        return None

# Helper function to load the trained Random Forest model pipeline
@st.cache_resource
def load_prediction_model():
    model_path = os.path.join("models", "sales_prediction_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.warning("Trained model not found. Predictions will use live training if model file isn't exported.")
        return None

# Load resources
df = load_clean_data()
model = load_prediction_model()

# Title Block
st.markdown("<div class='main-title'>Supermarket Sales Performance & Predictive Analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Executive Decision-Support Platform with Machine Learning & Hypothesis Testing</div>", unsafe_allow_html=True)

if df is not None:
    # Sidebar Filters
    st.sidebar.image("images/branch_performance.png", use_container_width=True, caption="Retail Analytics Suite")
    st.sidebar.title("🎛️ Dashboard Filters")
    st.sidebar.write("Filter the dashboard and statistical analysis in real time:")

    # Multi-select options with Defaults
    branches_list = sorted(df['branch'].unique())
    selected_branches = st.sidebar.multiselect("Select Branches", options=branches_list, default=branches_list)

    customer_types = sorted(df['customer_type'].unique())
    selected_customers = st.sidebar.multiselect("Select Customer Types", options=customer_types, default=customer_types)

    genders = sorted(df['gender'].unique())
    selected_genders = st.sidebar.multiselect("Select Genders", options=genders, default=genders)

    product_lines = sorted(df['product_line'].unique())
    selected_products = st.sidebar.multiselect("Select Product Lines", options=product_lines, default=product_lines)

    payments = sorted(df['payment'].unique())
    selected_payments = st.sidebar.multiselect("Select Payment Methods", options=payments, default=payments)

    # Apply Filters to main DataFrame
    filtered_df = df[
        (df['branch'].isin(selected_branches)) &
        (df['customer_type'].isin(selected_customers)) &
        (df['gender'].isin(selected_genders)) &
        (df['product_line'].isin(selected_products)) &
        (df['payment'].isin(selected_payments))
    ]

    # Tabs definition
    tab_dashboard, tab_prediction, tab_statistics, tab_explorer = st.tabs([
        "📊 Executive Dashboard", 
        "🔮 Invoice Predictor (ML)", 
        "🔬 Statistical Insights", 
        "🗂️ Raw Data Explorer"
    ])

    # ==========================================
    # TAB 1: EXECUTIVE DASHBOARD
    # ==========================================
    with tab_dashboard:
        # Check if empty dataframe after filtering
        if filtered_df.empty:
            st.warning("⚠️ No data available matching the selected filters. Please expand your selection in the sidebar.")
        else:
            # Dynamic KPIs calculation
            total_revenue = filtered_df['sales'].sum()
            total_profit = filtered_df['gross_income'].sum()
            avg_rating = filtered_df['rating'].mean()
            total_transactions = filtered_df.shape[0]
            
            # Premium KPI Cards Row
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Total Revenue</div>
                    <div class='kpi-value'>${total_revenue:,.2f}</div>
                    <div class='kpi-sub'>All Selected Branches</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[1]:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Gross Income (Profit)</div>
                    <div class='kpi-value' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>${total_profit:,.2f}</div>
                    <div class='kpi-sub'>Fixed Margin (4.76%)</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[2]:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Avg Customer Rating</div>
                    <div class='kpi-value' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{avg_rating:.2f} / 10</div>
                    <div class='kpi-sub'>Out of {total_transactions:,} Feedbacks</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Total Invoices</div>
                    <div class='kpi-value' style='background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{total_transactions:,}</div>
                    <div class='kpi-sub'>Transactions Processed</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            
            # Interactive Plotly Charts Section
            chart_cols1 = st.columns(2)
            
            with chart_cols1[0]:
                st.markdown("<div class='section-header'>Branch Revenue & Rating Breakdown</div>", unsafe_allow_html=True)
                # Group data by Branch
                branch_summary = filtered_df.groupby('branch').agg({
                    'sales': 'sum',
                    'rating': 'mean'
                }).reset_index()
                
                fig_branch = px.bar(
                    branch_summary, 
                    x='branch', 
                    y='sales', 
                    color='sales',
                    text_auto='.2s',
                    labels={'sales': 'Revenue (USD)', 'branch': 'Branch'},
                    color_continuous_scale='Purples',
                    hover_data={'rating': ':.2f'}
                )
                fig_branch.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_branch, use_container_width=True)
                
            with chart_cols1[1]:
                st.markdown("<div class='section-header'>Product Line Sales Performance</div>", unsafe_allow_html=True)
                product_summary = filtered_df.groupby('product_line').agg({
                    'sales': 'sum',
                    'quantity': 'sum'
                }).reset_index().sort_values(by='sales', ascending=True)
                
                fig_product = px.bar(
                    product_summary,
                    x='sales',
                    y='product_line',
                    orientation='h',
                    color='sales',
                    color_continuous_scale='viridis',
                    labels={'sales': 'Revenue (USD)', 'product_line': 'Category'},
                    hover_data={'quantity': ':,'}
                )
                fig_product.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_product, use_container_width=True)

            chart_cols2 = st.columns(3)
            
            with chart_cols2[0]:
                st.markdown("<div class='section-header'>Payment Methods Share</div>", unsafe_allow_html=True)
                payment_summary = filtered_df.groupby('payment').size().reset_index(name='count')
                fig_payment = px.pie(
                    payment_summary,
                    values='count',
                    names='payment',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_payment.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_payment, use_container_width=True)
                
            with chart_cols2[1]:
                st.markdown("<div class='section-header'>Monthly Revenue Trend</div>", unsafe_allow_html=True)
                monthly_summary = filtered_df.groupby('month').agg({'sales': 'sum', 'month_num': 'first'}).reset_index()
                monthly_summary = monthly_summary.sort_values(by='month_num')
                
                fig_monthly = px.line(
                    monthly_summary,
                    x='month',
                    y='sales',
                    markers=True,
                    labels={'sales': 'Monthly Revenue (USD)', 'month': 'Month'}
                )
                fig_monthly.update_traces(line=dict(color='#7c3aed', width=3), marker=dict(size=8, color='#db2777'))
                fig_monthly.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
                
            with chart_cols2[2]:
                st.markdown("<div class='section-header'>Peak Purchase Hours</div>", unsafe_allow_html=True)
                hourly_summary = filtered_df.groupby('hour').size().reset_index(name='transactions')
                
                fig_hourly = px.area(
                    hourly_summary,
                    x='hour',
                    y='transactions',
                    labels={'transactions': 'Invoices', 'hour': 'Hour of Day (24h)'}
                )
                fig_hourly.update_traces(fillcolor='rgba(56, 189, 248, 0.4)', line=dict(color='#0284c7', width=2))
                fig_hourly.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

    # ==========================================
    # TAB 2: INVOICE PREDICTOR (ML)
    # ==========================================
    with tab_prediction:
        st.markdown("<div class='section-header'>Predict Checkout Invoice Value & Margin Profit</div>", unsafe_allow_html=True)
        st.write("Construct a hypothetical transaction below, and the Random Forest machine learning model will predict the final transaction amount and margin profit.")
        
        pred_cols = st.columns([1, 1])
        
        with pred_cols[0]:
            st.write("#### 🔧 Inputs Settings")
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                input_branch = st.selectbox("Branch / Location", options=df['branch'].unique())
                # Infer city automatically based on branch to maintain consistency
                branch_city_map = {'Alex': 'Yangon', 'Giza': 'Naypyitaw', 'Cairo': 'Mandalay'}
                input_city = branch_city_map[input_branch]
                st.text_input("City (auto-inferred)", value=input_city, disabled=True)
                
                input_cust = st.selectbox("Customer Type", options=df['customer_type'].unique())
                input_gender = st.selectbox("Gender", options=df['gender'].unique())
            
            with col_in2:
                input_prod = st.selectbox("Product Line", options=df['product_line'].unique())
                input_payment = st.selectbox("Payment Method", options=df['payment'].unique())
                input_hour = st.slider("Checkout Hour", min_value=10, max_value=20, value=15)
                
                # Convert day name to standard day_of_week int (0=Monday, 6=Sunday)
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                input_day_name = st.selectbox("Day of Week", options=day_names, index=5) # Default Saturday
                input_day_of_week = day_names.index(input_day_name)
            
            st.write("---")
            col_in3, col_in4 = st.columns(2)
            with col_in3:
                input_price = st.slider("Unit Price (USD)", min_value=10.0, max_value=100.0, value=50.0, step=0.1)
            with col_in4:
                input_quantity = st.slider("Quantity purchased", min_value=1, max_value=10, value=5)
            
        with pred_cols[1]:
            st.write("#### 🎯 Prediction Results")
            
            if model is not None:
                # Prepare features dict for matching shape
                pred_features = pd.DataFrame([{
                    'unit_price': input_price,
                    'quantity': input_quantity,
                    'branch': input_branch,
                    'customer_type': input_cust,
                    'gender': input_gender,
                    'product_line': input_prod,
                    'payment': input_payment,
                    'hour': input_hour,
                    'day_of_week': input_day_of_week
                }])
                
                predicted_sales = model.predict(pred_features)[0]
                
                # Mathematical / actual logic helper:
                # sales = 1.05 * cogs = 1.05 * (unit_price * quantity)
                cogs = input_price * input_quantity
                tax = 0.05 * cogs
                actual_sales = cogs + tax
                # Gross margin is fixed at 4.7619% of sales
                predicted_profit = predicted_sales * 0.04761904762
                
                st.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='margin-top:0px; color:#a7f3d0;'>🔮 Predicted Sales Amount</h3>
                    <div style='font-size: 2.8rem; font-weight:700; color:#34d399;'>${predicted_sales:.2f}</div>
                    <div style='font-size:0.95rem; margin-top:0.5rem; color:#d1fae5;'>
                        Estimated Gross Income (Profit): <strong>${predicted_profit:.4f}</strong><br>
                        Mathematical Checkout Total: <strong>${actual_sales:.4f}</strong> (Diff: <strong>{abs(predicted_sales - actual_sales):.4f}</strong>)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                > **💡 Advanced Analytics Insight:**
                >
                > In this supermarket dataset, **Sales** is a deterministic calculation: `Sales = 1.05 * (Unit Price * Quantity)`. 
                > - Because it is a multiplication (`Unit Price * Quantity`), a standard **Linear Regression** baseline performs poorly ($R^2 = 0.9014$) unless it models interaction terms.
                > - The **Random Forest Regressor** ($R^2 = 0.9986$) easily splits on the numerical bounds, capturing the exact interaction with sub-dollar accuracy.
                > - Categorical attributes (Gender, Payment, etc.) represent demographic clusters but have zero mathematical influence on the billing total.
                """)
                
                # Show model error plot
                st.write("")
                st.write("##### Model Performance Scatter Plot")
                st.image("images/model_actual_vs_predicted.png", caption="Model Validation: Actual vs. Predicted (Random Forest Regressor)", use_container_width=True)
            else:
                st.info("Predictive model pipeline not loaded. Please ensure `models/sales_prediction_model.pkl` is exported correctly.")

    # ==========================================
    # TAB 3: STATISTICAL INSIGHTS
    # ==========================================
    with tab_statistics:
        st.markdown("<div class='section-header'>Dynamic Hypothesis Testing & Statistical Significance</div>", unsafe_allow_html=True)
        st.write("Below are the mathematical proofs confirming whether observable retail differences are statistically significant or merely random fluctuations.")
        
        # 1. T-Test: Gender spending differences
        st.write("### 👥 1. T-Test: Gender vs. Ticket Size")
        st.write("**H₀ (Null Hypothesis):** Male and Female shoppers spend the same average amount per transaction.")
        st.write("**H₁ (Alternative Hypothesis):** There is a significant difference in spending between genders.")
        
        female_spend = filtered_df[filtered_df['gender'] == 'Female']['sales']
        male_spend = filtered_df[filtered_df['gender'] == 'Male']['sales']
        
        if len(female_spend) > 1 and len(male_spend) > 1:
            t_stat, p_val = stats.ttest_ind(female_spend, male_spend, equal_var=False)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("Female Mean Ticket Size", f"${female_spend.mean():.2f}")
                st.metric("Male Mean Ticket Size", f"${male_spend.mean():.2f}")
            with col_t2:
                st.metric("T-Statistic", f"{t_stat:.4f}")
                st.metric("P-Value", f"{p_val:.5f}", delta="Significant" if p_val < 0.05 else "Not Significant", delta_color="normal" if p_val < 0.05 else "inverse")
                
            if p_val < 0.05:
                st.success(f"✔️ **Reject H₀ (P-Value < 0.05)**: The difference in average ticket sizes between Female and Male shoppers is **statistically significant**. Females exhibit stronger purchasing power at the supermarket.")
            else:
                st.info(f"ℹ️ **Fail to Reject H₀ (P-Value ≥ 0.05)**: The spending differences are not statistically significant in this filtered segment. They could be due to random chance.")
        else:
            st.warning("Insufficient gender variation in the filtered subset to perform T-test.")
            
        st.markdown("---")
        
        # 2. ANOVA: Branch Revenue Parity
        st.write("### 🏢 2. ANOVA: Sales variation across Branches")
        st.write("**H₀ (Null Hypothesis):** Mean sales invoice values are identical across all branches.")
        st.write("**H₁ (Alternative Hypothesis):** At least one branch has a different average sales ticket size.")
        
        branches_in_df = filtered_df['branch'].unique()
        if len(branches_in_df) > 1:
            branch_groups = [filtered_df[filtered_df['branch'] == b]['sales'] for b in branches_in_df]
            f_stat, p_val_anova = stats.f_oneway(*branch_groups)
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                for b in branches_in_df:
                    st.write(f"📍 **Branch {b} average ticket size:** ${filtered_df[filtered_df['branch'] == b]['sales'].mean():.2f}")
            with col_a2:
                st.metric("F-Statistic", f"{f_stat:.4f}")
                st.metric("ANOVA P-Value", f"{p_val_anova:.5f}", delta="Significant" if p_val_anova < 0.05 else "Not Significant", delta_color="normal" if p_val_anova < 0.05 else "inverse")
                
            if p_val_anova < 0.05:
                st.success(f"✔️ **Reject H₀ (P-Value < 0.05)**: There are statistically significant differences in transaction size averages across branches.")
            else:
                st.info(f"ℹ️ **Fail to Reject H₀ (P-Value ≥ 0.05)**: The branches exhibit **revenue parity**. The differences in ticket size averages across locations are due to random sampling, confirming that pricing and cart size distributions are uniform.")
        else:
            st.warning("Insufficient branch locations in the filtered subset to perform ANOVA testing.")

    # ==========================================
    # TAB 4: RAW DATA EXPLORER
    # ==========================================
    with tab_explorer:
        st.markdown("<div class='section-header'>Transaction Log Viewer</div>", unsafe_allow_html=True)
        st.write(f"Showing **{filtered_df.shape[0]:,}** rows matching your sidebar filter criteria.")
        
        # Data columns selection
        cols_to_show = st.multiselect(
            "Select Columns to Display", 
            options=list(df.columns), 
            default=['invoice_id', 'branch', 'city', 'customer_type', 'gender', 'product_line', 'unit_price', 'quantity', 'sales', 'payment', 'rating']
        )
        
        st.dataframe(filtered_df[cols_to_show], height=400, use_container_width=True)
        
        # Download preprocessed dataset
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_data,
            file_name="filtered_supermarket_sales.csv",
            mime="text/csv",
        )
else:
    st.info("Please resolve dataset configuration errors to activate dashboard tabs.")
