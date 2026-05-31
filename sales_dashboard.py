# ============================================================
#  PROJECT 1: SALES PERFORMANCE DASHBOARD
#  Tools: Python, Pandas, Matplotlib, Seaborn
#  Dataset: Superstore Sales (download from Kaggle)
#  Author: [Harsh Jha]
# ============================================================

# STEP 1: IMPORT LIBRARIES
# These are the tools we need. Install them once using:
# pip install pandas matplotlib seaborn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================
# Download "Sample - Superstore.csv" from Kaggle:
# https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
#
# Then put it in the same folder as this file and run the script.
# parse_dates converts the date column from text to actual dates.

df = pd.read_csv('superstore_sales.csv', parse_dates=['Order Date'], encoding='latin1')

# Quick check — see first 5 rows and column names
print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())


# ============================================================
# STEP 3: CLEAN THE DATA
# ============================================================

# Check for missing values in each column
print("\nMissing values:\n", df.isnull().sum())

# Drop any rows where Sales or Profit is missing
df.dropna(subset=['Sales', 'Profit'], inplace=True)

# Create a Revenue column (Sales × Quantity)
df['Revenue'] = df['Sales'] * df['Quantity']

# Extract Month for grouping (e.g. "Jan 2022")
df['Month'] = df['Order Date'].dt.to_period('M')
df['Month_str'] = df['Order Date'].dt.strftime('%b %Y')

print("\nCleaned data shape:", df.shape)


# ============================================================
# STEP 4: ANALYSE THE DATA
# ============================================================

# --- 4a. KPI Summary ---
total_revenue = df['Revenue'].sum()
total_profit  = df['Profit'].sum()
total_orders  = len(df)
profit_margin = (total_profit / total_revenue) * 100

print(f"\n=== KPI SUMMARY ===")
print(f"Total Revenue  : ₹{total_revenue:,.0f}")
print(f"Total Profit   : ₹{total_profit:,.0f}")
print(f"Total Orders   : {total_orders:,}")
print(f"Profit Margin  : {profit_margin:.1f}%")

# --- 4b. Monthly Revenue Trend ---
monthly_revenue = (
    df.groupby('Month')['Revenue']
    .sum()
    .reset_index()
)
monthly_revenue['Month_str'] = monthly_revenue['Month'].dt.strftime('%b %Y')
print("\nMonthly Revenue (first 5):")
print(monthly_revenue.head())

# --- 4c. Revenue by Region ---
region_revenue = (
    df.groupby('Region')['Revenue']
    .sum()
    .sort_values(ascending=True)  # ascending for horizontal bar chart
)
print("\nRevenue by Region:")
print(region_revenue)

# --- 4d. Profit by Category ---
category_profit = (
    df.groupby('Category')['Profit']
    .sum()
    .sort_values(ascending=False)
)
print("\nProfit by Category:")
print(category_profit)

# --- 4e. Top 10 Products by Revenue ---
top_products = (
    df.groupby('Product Name')['Revenue']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print("\nTop 10 Products:")
print(top_products)

# --- 4f. Discount Impact on Profit ---
discount_profit = (
    df.groupby('Discount')['Profit']
    .mean()
)
print("\nAvg Profit by Discount Level:")
print(discount_profit)


# ============================================================
# STEP 5: VISUALISE — BUILD THE DASHBOARD
# ============================================================

# Color palette — clean and professional
COLORS = {
    'blue':   '#378ADD',
    'teal':   '#1D9E75',
    'purple': '#7F77DD',
    'coral':  '#D85A30',
    'bg':     '#F8F8F8',
    'card':   '#FFFFFF',
    'text':   '#2C2C2A',
    'muted':  '#888780'
}

# Global style settings
plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'font.size':            10,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'figure.facecolor':     COLORS['bg'],
    'axes.facecolor':       COLORS['card'],
    'text.color':           COLORS['text'],
    'axes.labelcolor':      COLORS['text'],
    'xtick.color':          COLORS['muted'],
    'ytick.color':          COLORS['muted'],
})

# Create a large figure (18 wide × 14 tall inches)
fig = plt.figure(figsize=(18, 14), facecolor=COLORS['bg'])

# Main title
fig.suptitle(
    'Superstore Sales Performance Dashboard',
    fontsize=22, fontweight='bold',
    color=COLORS['text'], y=0.97, x=0.5
)


# ------ KPI CARDS (4 boxes at the top) ------
kpis = [
    ('Total Revenue',  f'₹{total_revenue:,.0f}',  COLORS['blue']),
    ('Total Profit',   f'₹{total_profit:,.0f}',   COLORS['teal']),
    ('Total Orders',   f'{total_orders:,}',         COLORS['purple']),
    ('Profit Margin',  f'{profit_margin:.1f}%',     COLORS['coral']),
]
for i, (label, val, color) in enumerate(kpis):
    ax_kpi = fig.add_axes([0.04 + i * 0.24, 0.87, 0.20, 0.055])
    ax_kpi.set_facecolor(COLORS['card'])
    for spine in ax_kpi.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor('#D3D1C7')
    ax_kpi.set_xticks([])
    ax_kpi.set_yticks([])
    ax_kpi.text(0.5, 0.72, label, ha='center', va='center',
                transform=ax_kpi.transAxes, fontsize=9.5, color=COLORS['muted'])
    ax_kpi.text(0.5, 0.25, val, ha='center', va='center',
                transform=ax_kpi.transAxes, fontsize=18, fontweight='bold', color=color)

# ------ CHART 1: Monthly Revenue Trend (Line Chart) ------
ax1 = fig.add_axes([0.04, 0.54, 0.56, 0.29])
ax1.set_facecolor(COLORS['card'])

x = range(len(monthly_revenue))

# Shaded area under the line
ax1.fill_between(x, monthly_revenue['Revenue'], alpha=0.15, color=COLORS['blue'])

# Line with circle markers
ax1.plot(
    x, monthly_revenue['Revenue'],
    color=COLORS['blue'], linewidth=2.5,
    marker='o', markersize=5,
    markerfacecolor='white', markeredgewidth=2
)

ax1.set_xticks(x)
ax1.set_xticklabels(monthly_revenue['Month_str'], rotation=45, ha='right', fontsize=8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v/1000:.0f}K'))
ax1.set_title('Monthly Revenue Trend', fontsize=13, fontweight='bold',
              color=COLORS['text'], pad=12, loc='left')
ax1.yaxis.grid(True, color='#EBEBEB', linewidth=0.5)
ax1.tick_params(axis='both', length=0)
for spine in ['left', 'bottom']:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_color('#D3D1C7')
    ax1.spines[spine].set_linewidth(0.5)

# ------ CHART 2: Revenue by Region (Horizontal Bar) ------
ax2 = fig.add_axes([0.67, 0.54, 0.29, 0.29])
ax2.set_facecolor(COLORS['card'])

bar_colors = [COLORS['blue'], COLORS['teal'], COLORS['purple'], COLORS['coral']]
bars2 = ax2.barh(
    region_revenue.index, region_revenue.values,
    color=bar_colors, height=0.55, edgecolor='none'
)
# Add value labels at end of each bar
for bar, val in zip(bars2, region_revenue.values):
    ax2.text(
        bar.get_width() + region_revenue.max() * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f'₹{val/1000:.0f}K', va='center', fontsize=9, color=COLORS['text']
    )

ax2.set_xlim(0, region_revenue.max() * 1.2)
ax2.set_title('Revenue by Region', fontsize=13, fontweight='bold',
              color=COLORS['text'], pad=12, loc='left')
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.tick_params(axis='y', length=0, labelsize=10)

# ------ CHART 3: Top 10 Products (Horizontal Bar) ------
ax3 = fig.add_axes([0.04, 0.13, 0.42, 0.34])
ax3.set_facecolor(COLORS['card'])

# Reverse so highest bar is at the top
ax3.barh(
    top_products.index[::-1], top_products.values[::-1],
    color=COLORS['blue'], height=0.65, edgecolor='none', alpha=0.85
)
ax3.set_title('Top 10 Products by Revenue', fontsize=13, fontweight='bold',
              color=COLORS['text'], pad=12, loc='left')
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v/1000:.0f}K'))
for spine in ax3.spines.values():
    spine.set_visible(False)
ax3.tick_params(axis='both', length=0, labelsize=9)
ax3.xaxis.grid(True, color='#EBEBEB', linewidth=0.5)

# ------ CHART 4: Profit by Category (Pie Chart) ------
ax4 = fig.add_axes([0.54, 0.13, 0.19, 0.34])
ax4.set_facecolor(COLORS['card'])

wedges, texts, autotexts = ax4.pie(
    category_profit.values,
    labels=category_profit.index,
    autopct='%1.1f%%',
    colors=[COLORS['blue'], COLORS['teal'], COLORS['purple']],
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'edgecolor': COLORS['card'], 'linewidth': 2}
)
for t in texts:
    t.set_fontsize(9)
    t.set_color(COLORS['muted'])
for at in autotexts:
    at.set_fontsize(9)
    at.set_color(COLORS['card'])
    at.set_fontweight('bold')

ax4.set_title('Profit by\nCategory', fontsize=13, fontweight='bold',
              color=COLORS['text'], pad=8, loc='left')

# ------ CHART 5: Discount vs Avg Profit (Bar Chart) ------
ax5 = fig.add_axes([0.76, 0.13, 0.21, 0.34])
ax5.set_facecolor(COLORS['card'])

bar_disc_colors = [
    COLORS['teal'] if v > 0 else COLORS['coral']
    for v in discount_profit.values
]
ax5.bar(
    discount_profit.index.astype(str),
    discount_profit.values,
    color=bar_disc_colors, width=0.5, edgecolor='none'
)
ax5.set_title('Avg Profit by\nDiscount Level', fontsize=13, fontweight='bold',
              color=COLORS['text'], pad=8, loc='left')
ax5.set_xlabel('Discount', fontsize=9, color=COLORS['muted'])
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v:.0f}'))
for spine in ax5.spines.values():
    spine.set_visible(False)
ax5.tick_params(axis='both', length=0, labelsize=9)
ax5.yaxis.grid(True, color='#EBEBEB', linewidth=0.5)

# ------ FOOTER ------
fig.text(
    0.5, 0.04,
    'Data Source: Superstore Sales Dataset  |  Analysis by: [Harsh Jha]  |  Tools: Python · Pandas · Matplotlib',
    ha='center', fontsize=9, color=COLORS['muted']
)


# ============================================================
# STEP 6: SAVE THE DASHBOARD
# ============================================================
plt.savefig(
    'sales_dashboard.png',
    dpi=150,
    bbox_inches='tight',
    facecolor=COLORS['bg'],
    edgecolor='none'
)
print("\n✅ Dashboard saved as 'sales_dashboard.png'")

