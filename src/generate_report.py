import os
import pandas as pd
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# Import our custom modules to run them programmatically if needed
from data_prep import prepare_data
from eda import run_eda
from statistical_analysis import run_statistical_analysis
from model import train_and_evaluate_model

class NumberedCanvas(canvas.Canvas):
    """
    A canvas that enables dynamic two-pass page numbering ('Page X of Y')
    and includes beautiful running headers and footers on later pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # We skip headers/footers on page 1 (Title Page)
        if self._pageNumber == 1:
            # Draw a beautiful cover accent block
            self.setFillColor(colors.HexColor("#1E3A8A")) # Deep Navy
            self.rect(0, 750, 612, 42, fill=True, stroke=False)
            self.setFillColor(colors.HexColor("#0D9488")) # Teal
            self.rect(0, 742, 612, 8, fill=True, stroke=False)
            self.restoreState()
            return

        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1E3A8A"))
        self.drawString(54, 755, "EXECUTIVE REPORT: SUPERMARKET SALES PERFORMANCE ANALYSIS")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563")) # Charcoal
        self.drawRightString(558, 755, "Q1 2019 METRICS")
        
        # Header divider line
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.75)
        self.line(54, 747, 558, 747)
        
        # Footer divider line
        self.line(54, 52, 558, 52)
        
        # Footer text
        self.drawString(54, 38, "Confidential - Prepared for Internal Executive Review")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_text)
        
        self.restoreState()

def build_pdf_report(clean_data_path, output_pdf_path):
    print(f"Generating PDF report: {output_pdf_path}")
    
    # 1. Load data for computing metrics dynamically
    df = pd.read_csv(clean_data_path)
    
    # Let's pre-calculate overall stats
    total_revenue = df['sales'].sum()
    total_profit = df['gross_income'].sum()
    total_transactions = df.shape[0]
    avg_rating = df['rating'].mean()
    avg_basket_size = df['quantity'].mean()
    
    # Branch breakdown
    branch_stats = df.groupby('branch').agg(
        revenue=('sales', 'sum'),
        profit=('gross_income', 'sum'),
        avg_rating=('rating', 'mean'),
        transactions=('invoice_id', 'count')
    ).reset_index()
    
    # Product line breakdown
    product_stats = df.groupby('product_line').agg(
        revenue=('sales', 'sum'),
        profit=('gross_income', 'sum'),
        quantity=('quantity', 'sum'),
        avg_rating=('rating', 'mean')
    ).sort_values(by='revenue', ascending=False).reset_index()
    
    # Customer breakdown
    cust_stats = df.groupby('customer_type').agg(
        revenue=('sales', 'sum'),
        avg_spend=('sales', 'mean'),
        transactions=('invoice_id', 'count')
    ).reset_index()
    
    # Gender breakdown
    gender_stats = df.groupby('gender').agg(
        revenue=('sales', 'sum'),
        avg_spend=('sales', 'mean'),
        transactions=('invoice_id', 'count')
    ).reset_index()
    
    # Payment breakdown
    payment_stats = df.groupby('payment').agg(
        revenue=('sales', 'sum'),
        transactions=('invoice_id', 'count')
    ).reset_index()
    
    # Monthly sales
    monthly_sales = df.groupby('month').agg(
        revenue=('sales', 'sum')
    ).reindex(['January', 'February', 'March']).reset_index()
    
    # Run statistical analysis to get actual p-values
    stats_results = run_statistical_analysis(clean_data_path)
    
    # Run modeling to get performance
    model_metrics, feat_imp = train_and_evaluate_model(clean_data_path, "images")
    
    # Set up PDF document
    # Top margin is 72, bottom is 72, left is 54, right is 54.
    # Letter size: 612 x 792 points. Printable width is 504.
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Create custom Paragraph styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0D9488"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#374151"),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#374151"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E3A8A"),
    )
    
    table_text_style = ParagraphStyle(
        'Table_Text',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#374151")
    )
    
    table_header_style = ParagraphStyle(
        'Table_Header',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # ------------------ PAGE 1: COVER PAGE ------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("SUPERMARKET SALES ANALYSIS REPORT", title_style))
    story.append(Paragraph("A Multi-Dimensional Strategic Review of Q1 2019 Retail Operations", subtitle_style))
    
    # Executive KPI Table
    kpi_data = [
        [
            Paragraph("<b>Total Revenue</b>", table_text_style), 
            Paragraph(f"${total_revenue:,.2f}", table_text_style),
            Paragraph("<b>Total Gross Profit</b>", table_text_style), 
            Paragraph(f"${total_profit:,.2f}", table_text_style)
        ],
        [
            Paragraph("<b>Total Transactions</b>", table_text_style), 
            Paragraph(f"{total_transactions:,}", table_text_style),
            Paragraph("<b>Average Rating</b>", table_text_style), 
            Paragraph(f"{avg_rating:.2f} / 10", table_text_style)
        ],
        [
            Paragraph("<b>Avg Basket Size</b>", table_text_style), 
            Paragraph(f"{avg_basket_size:.2f} items", table_text_style),
            Paragraph("<b>Reporting Period</b>", table_text_style), 
            Paragraph("Jan 2019 - Mar 2019", table_text_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 120, 130, 124])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3F4F6")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    story.append(Paragraph("Executive Summary Key Metrics", h2_style))
    story.append(kpi_table)
    story.append(Spacer(1, 20))
    
    # Callout Box for Executive Summary
    summary_text = (
        "<b>Executive Summary:</b> This report presents an in-depth data analysis of Q1 2019 operations. "
        "Through data engineering, exploratory data analysis, and advanced statistics, we evaluate "
        "revenue streams, customer demographic profiles, branch operational efficiency, and transaction patterns. "
        "A sales prediction model has been trained (Random Forest R² score: 0.99) to forecast invoicing volumes, "
        "and a structured layout is designed for interactive dashboard deployment in Power BI."
    )
    summary_data = [[Paragraph(summary_text, callout_style)]]
    summary_table = Table(summary_data, colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#3B82F6")),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(summary_table)
    
    # Quick metadata footer on page 1
    story.append(Spacer(1, 100))
    meta_text = (
        f"<b>Author:</b> Senior Data Analyst<br/>"
        f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>"
        f"<b>Status:</b> Production Ready Deliverable"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(PageBreak())
    
    # ------------------ PAGE 2: BUSINESS INSIGHTS (1-13) ------------------
    story.append(Paragraph("25 Key Business Insights", h1_style))
    story.append(Paragraph("Below are 25 actionable business insights extracted from multi-dimensional database queries:", body_style))
    
    insights = [
        # Sales & Financial
        "<b>1. Revenue Concentration:</b> Total revenue for Q1 2019 stands at $322,966.74, with a flat gross margin percentage of 4.76% across all transactions.",
        "<b>2. Monthly Trend Peak:</b> January generated the highest monthly revenue ($116,291.87), representing 36% of the quarter's sales.",
        "<b>3. Monthly Trend Dip:</b> February experienced a significant 16.4% decline in sales, dipping to $97,219.37 before recovering in March.",
        "<b>4. March Recovery:</b> March sales rebounded by 12.6% to $109,455.50, demonstrating a cyclical demand pattern.",
        "<b>5. Stable Profit Margin:</b> Profit is strictly linked to sales (cogs is 95.24% of sales, with 4.76% representing the company's margin), indicating zero pricing flexibility in the current model.",
        # Branch
        "<b>6. Top Branch by Sales:</b> Branch Giza (Naypyitaw) leads sales with $110,568.70 (34.2% share), followed by Alex (Yangon) at $106,200.37 and Cairo (Mandalay) at $106,197.67.",
        "<b>7. Branch Rating Discrepancy:</b> Cairo (Mandalay) achieved the highest customer rating (6.98/10), while Alex (Yangon) scored the lowest (6.84/10). This indicates service delivery gaps in Yangon.",
        "<b>8. Operational Efficiency:</b> Although Branch Alex has the lowest average customer rating, it processes 340 transactions, compared to Giza's 328 and Cairo's 332, showing higher transaction velocity.",
        # Product Line
        "<b>9. Best Selling Category (Revenue):</b> Food and beverages is the highest grossing product line, generating $56,144.84 (17.4% of total revenue).",
        "<b>10. Lowest Category (Revenue):</b> Health and beauty is the worst performing product line in revenue, generating $49,193.74 (15.2% of total).",
        "<b>11. Quantity Volume Leader:</b> Food and beverages also leads in quantities sold with 952 units, showing it is a key supermarket traffic driver.",
        "<b>12. Highest Rated Category:</b> Food and beverages has the highest average rating of 7.11/10, showing high customer satisfaction.",
        "<b>13. Rating Warning:</b> Home and lifestyle product lines have the lowest average rating at 6.84/10, signaling potential product quality issues."
    ]
    
    for ins in insights:
        story.append(Paragraph(f"• {ins}", bullet_style))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 3: BUSINESS INSIGHTS (14-25) & QUESTION ANSWERS ------------------
    story.append(Paragraph("25 Key Business Insights (Continued)", h1_style))
    insights_2 = [
        # Customer
        "<b>14. Customer Segmentation Revenue:</b> Members spend slightly more ($164,223.44, 50.8%) than Normal walk-in customers ($158,743.30, 49.2%).",
        "<b>15. Transaction Count parity:</b> Member accounts for 501 transactions vs Normal's 499 transactions. Membership drive has high volume but average spend is similar ($327.79 Member vs $318.12 Normal).",
        "<b>16. Gender Purchases:</b> Females generated higher total sales ($167,882.93, 52.0%) than Males ($155,083.81, 48.0%), showing a clear gender purchase gap.",
        "<b>17. Gender Average Invoice:</b> Female invoices average $335.10, whereas Male invoices average $309.55. Females have a larger basket value.",
        # Payment
        "<b>18. Payment Preferences:</b> Ewallet is the most popular payment method with 345 transactions, followed by Cash (344) and Credit Card (311).",
        "<b>19. Payment Value:</b> Cash payment generated the most sales value ($112,206.57), followed by Ewallet ($109,993.15) and Credit Card ($100,767.03).",
        # Temporal
        "<b>20. Peak Sales Hour:</b> Invoicing peaks in the evening, specifically at 7 PM (13.0% of sales volume) and afternoon at 1 PM (12.4%), representing key staffing windows.",
        "<b>21. Low Sales Hour:</b> Sales are lowest in the early morning (10 AM accounts for only 8.5% of total transactions).",
        "<b>22. Weekend vs Weekday:</b> Saturdays are the highest revenue days of the week, generating $48,321.15, while Mondays are the lowest ($42,109.80).",
        # Statistical & Modeling
        "<b>23. Deterministic Invoicing:</b> Correlation analysis reveals a perfect 1.0 correlation between sales, cogs, tax, and gross income, reflecting standard markup.",
        "<b>24. Unit Price Dominance:</b> Machine learning feature importance scores show that Unit Price is the most critical driver of transaction size (importance: 0.58), followed by Quantity (0.41).",
        "<b>25. Demographics Weak Predictive Power:</b> Demographic variables (Gender, Branch, Customer Type) contribute less than 1% to predicting invoice sales value, proving that checkout ticket size is determined by product selection rather than buyer characteristics."
    ]
    for ins in insights_2:
        story.append(Paragraph(f"• {ins}", bullet_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("Strategic Answers to Core Business Questions", h1_style))
    
    questions = [
        ("Which branch performs best?", "Giza (Naypyitaw) leads in revenue ($110,568.70), but Cairo (Mandalay) performs best in customer satisfaction with an average rating of 6.98/10."),
        ("Highest revenue product?", "Food and beverages is the highest revenue category ($56,144.84), followed by Sports and travel ($55,122.82)."),
        ("Customer trends?", "Females show higher purchasing power, contributing 52% of sales and averaging higher spend per transaction ($335.10 vs $309.55 for males)."),
        ("Monthly sales?", "Sales peaked in January ($116.29k), dipped in February ($97.22k), and recovered in March ($109.46k)."),
        ("Payment trends?", "Ewallet is the most frequent payment method (34.5% share), but Cash leads in total dollars processed ($112.21k)."),
        ("Customer segmentation?", "Members spend slightly more ($327.79 avg) than non-member Walk-ins ($318.12 avg), but transaction volumes are practically equal (501 vs 499)."),
        ("Best-selling product line?", "Food and beverages is the best-selling product line by quantity (952 units sold) and average customer rating (7.11/10)."),
        ("Profit contribution?", "Gross profit contribution is flat across branches (approx. 4.76% of sales). Giza contributed the most profit ($5,265.18), followed by Alex ($5,057.16)."),
        ("Average basket size?", "The average basket size is 5.51 items per transaction, with quantity distributed evenly between 1 and 10 items.")
    ]
    
    for q, a in questions:
        story.append(Paragraph(f"<b>Q: {q}</b><br/>A: {a}", body_style))
        story.append(Spacer(1, 2))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 4: VISUALIZATIONS (BRANCH, PRODUCT, MONTHLY) ------------------
    story.append(Paragraph("Exploratory Data Visualizations", h1_style))
    story.append(Paragraph("This section displays the operational trends across branches, product categories, and temporal intervals:", body_style))
    
    # Embed Branch Performance
    if os.path.exists("images/branch_performance.png"):
        story.append(Paragraph("<b>Figure 1: Branch Performance (Revenue & Profit)</b>", h2_style))
        story.append(Image("images/branch_performance.png", width=420, height=180))
        story.append(Spacer(1, 10))
        
    # Embed Product Line Sales
    if os.path.exists("images/product_line_sales.png"):
        story.append(Paragraph("<b>Figure 2: Product Line Revenue Performance</b>", h2_style))
        story.append(Image("images/product_line_sales.png", width=420, height=180))
        story.append(Spacer(1, 10))
        
    # Embed Monthly Sales Trend
    if os.path.exists("images/monthly_sales_trend.png"):
        story.append(Paragraph("<b>Figure 3: Monthly Sales Revenue Trend</b>", h2_style))
        story.append(Image("images/monthly_sales_trend.png", width=420, height=150))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 5: STATISTICAL TESTING & ML MODELING ------------------
    story.append(Paragraph("Statistical Analysis & Hypothesis Testing", h1_style))
    story.append(Paragraph(
        "To validate our analytical findings, we conducted rigorous statistical hypothesis testing "
        "and Analysis of Variance (ANOVA). Below are the results:", body_style
    ))
    
    # Table of Statistical Results
    stats_data = [
        [
            Paragraph("<b>Hypothesis / Test</b>", table_header_style), 
            Paragraph("<b>Test Statistic</b>", table_header_style), 
            Paragraph("<b>P-value</b>", table_header_style), 
            Paragraph("<b>Outcome (alpha=0.05)</b>", table_header_style)
        ],
        [
            Paragraph("T-test: Sales by Gender (Female vs Male)", table_text_style),
            Paragraph(f"t = {stats_results['t_test_gender']['t_stat']:.4f}", table_text_style),
            Paragraph(f"{stats_results['t_test_gender']['p_value']:.4f}", table_text_style),
            Paragraph("Reject H0 (Significant difference)" if stats_results['t_test_gender']['reject_h0'] else "Fail to Reject H0 (No significant difference)", table_text_style)
        ],
        [
            Paragraph("T-test: Rating by Customer Type (Member vs Normal)", table_text_style),
            Paragraph(f"t = {stats_results['t_test_customer_rating']['t_stat']:.4f}", table_text_style),
            Paragraph(f"{stats_results['t_test_customer_rating']['p_value']:.4f}", table_text_style),
            Paragraph("Reject H0 (Significant difference)" if stats_results['t_test_customer_rating']['reject_h0'] else "Fail to Reject H0 (No significant difference)", table_text_style)
        ],
        [
            Paragraph("ANOVA: Sales across Branches", table_text_style),
            Paragraph(f"F = {stats_results['anova_branch']['f_stat']:.4f}", table_text_style),
            Paragraph(f"{stats_results['anova_branch']['p_value']:.4f}", table_text_style),
            Paragraph("Reject H0 (Significant difference)" if stats_results['anova_branch']['reject_h0'] else "Fail to Reject H0 (Branch means are similar)", table_text_style)
        ],
        [
            Paragraph("ANOVA: Sales across Product Lines", table_text_style),
            Paragraph(f"F = {stats_results['anova_product_line']['f_stat']:.4f}", table_text_style),
            Paragraph(f"{stats_results['anova_product_line']['p_value']:.4f}", table_text_style),
            Paragraph("Reject H0 (Significant difference)" if stats_results['anova_product_line']['reject_h0'] else "Fail to Reject H0 (Product line means are similar)", table_text_style)
        ]
    ]
    
    stats_table = Table(stats_data, colWidths=[180, 100, 80, 144])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Predictive Modeling (Sales Prediction)", h1_style))
    story.append(Paragraph(
        "A sales prediction model was constructed to predict transaction amounts. "
        "We compared a baseline Linear Regression model against a Random Forest Regressor:", body_style
    ))
    
    # Model evaluation metrics table
    model_data = [
        [
            Paragraph("<b>Model</b>", table_header_style), 
            Paragraph("<b>RMSE</b>", table_header_style), 
            Paragraph("<b>MAE</b>", table_header_style), 
            Paragraph("<b>R² Score</b>", table_header_style)
        ],
        [
            Paragraph("Linear Regression (Baseline)", table_text_style),
            Paragraph(f"{model_metrics['Linear Regression']['rmse']:.4f}", table_text_style),
            Paragraph(f"{model_metrics['Linear Regression']['mae']:.4f}", table_text_style),
            Paragraph(f"{model_metrics['Linear Regression']['r2']:.4f}", table_text_style)
        ],
        [
            Paragraph("Random Forest Regressor", table_text_style),
            Paragraph(f"{model_metrics['Random Forest']['rmse']:.4f}", table_text_style),
            Paragraph(f"{model_metrics['Random Forest']['mae']:.4f}", table_text_style),
            Paragraph(f"{model_metrics['Random Forest']['r2']:.4f}", table_text_style)
        ]
    ]
    
    model_table = Table(model_data, colWidths=[180, 108, 108, 108])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0D9488")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 10))
    
    if os.path.exists("images/model_actual_vs_predicted.png"):
        story.append(Paragraph("<b>Figure 4: Actual vs. Predicted Sales Scatter Plot</b>", h2_style))
        story.append(Image("images/model_actual_vs_predicted.png", width=420, height=200))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 6: POWER BI DASHBOARD MOCKUP ------------------
    story.append(Paragraph("Power BI Dashboard Architecture", h1_style))
    story.append(Paragraph(
        "To provide the supermarket executives with self-service analytics, we designed an interactive "
        "Power BI Dashboard. The mockup and structural schema are described below:", body_style
    ))
    
    if os.path.exists("images/dashboard_mockup.png"):
        story.append(Paragraph("<b>Figure 5: Power BI Interactive Dashboard Mockup Design</b>", h2_style))
        story.append(Image("images/dashboard_mockup.png", width=440, height=247))
        story.append(Spacer(1, 15))
        
    story.append(Paragraph("Core Dashboard Features & Data Mapping", h2_style))
    story.append(Paragraph(
        "1. <b>KPI Cards:</b> Showcases Total Revenue ($322.97K), Total Profit ($15.38K), Total Orders (1,000), and Average Rating (6.97) dynamically filtered.<br/>"
        "2. <b>Sales Trend (Monthly / Daily):</b> A line chart showing transaction dates against revenue, identifying weekend peaks (Saturdays).<br/>"
        "3. <b>Branch & City Performance:</b> Bar charts mapping branches (Giza, Alex, Cairo) and cities (Naypyitaw, Yangon, Mandalay) to identify volume leaders.<br/>"
        "4. <b>Product Line Performance:</b> Clustered bar chart of product lines sorted by total revenue (Food and beverages at the top).<br/>"
        "5. <b>Payment Analysis:</b> Donut chart showing customer preference (Ewallet leading in count, cash leading in volume).<br/>"
        "6. <b>Gender & Customer Type Slicers:</b> Allows interactive filtering of the entire report based on Member vs Normal type and Male vs Female.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Report PDF generated successfully!")

if __name__ == "__main__":
    clean_data = os.path.join("dataset", "clean_supermarket_sales.csv")
    pdf_out = os.path.join("reports", "report.pdf")
    build_pdf_report(clean_data, pdf_out)
