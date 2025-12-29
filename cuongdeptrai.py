import pandas as pd
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# --- Bắt đầu code xử lý ---
print("⏳ Đang xử lý dữ liệu...") 
# =============================================================================
# 1. DATA LOADING & INITIAL EXPLORATION
# =============================================================================

print("\n🔍 STEP 1: DATA LOADING & EXPLORATION")
print("-" * 40)
datasets = {}

try:
    # 💡 LƯU Ý: Chỉ cần điền đúng tên file (vì file code và data nằm cùng thư mục)
    amazon_sales = pd.read_csv('Amazon Sale Report.csv')
    international_sales = pd.read_csv('International sale Report.csv')
    may_2022 = pd.read_csv('May-2022.csv')
    pl_march_2021 = pd.read_csv('P  L March 2021.csv') # Kiểm tra kỹ tên file này, có thể có ký tự lạ
    sale_report = pd.read_csv('Sale Report.csv')
    
    print("✅ All datasets loaded successfully!")
    
    # Dataset overview
    datasets = {
        'Amazon Sales': amazon_sales,
        'International Sales': international_sales,
        'May 2022 Pricing': may_2022,
        'P&L March 2021': pl_march_2021,
        'Sale Report': sale_report
    }
    
    print(f"\n📋 Dataset Overview:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
except FileNotFoundError as e:
    print(f"❌ Lỗi: Không tìm thấy file. Hãy chắc chắn file CSV nằm cùng thư mục với file code.")
    print(f"Chi tiết lỗi: {e}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    # =============================================================================
# 2. DATA PREPROCESSING & CLEANING
# =============================================================================

print("\n🧹 STEP 2: DATA PREPROCESSING & CLEANING")
print("-" * 40)

def analyze_missing_values(df, name):
    """Analyze missing values in dataset"""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Percentage', ascending=False)
    
    print(f"\n📊 Missing Values Analysis - {name}:")
    print(missing_df[missing_df['Missing Count'] > 0])
    return missing_df

# Analyze missing values for each dataset
missing_analysis = {}
for name, df in datasets.items():
    missing_analysis[name] = analyze_missing_values(df, name)

# Clean Amazon Sales Data (Primary dataset for analysis)
print("\n🔧 Cleaning Amazon Sales Data...")

# Create a copy for cleaning
amazon_clean = amazon_sales.copy()

# Handle missing values
print(f"Original Amazon dataset shape: {amazon_clean.shape}")

# Remove rows where critical columns are missing
critical_columns = ['Order ID', 'Date', 'Amount', 'Qty']
before_cleaning = len(amazon_clean)
amazon_clean = amazon_clean.dropna(subset=[col for col in critical_columns if col in amazon_clean.columns])
after_cleaning = len(amazon_clean)
print(f"Removed {before_cleaning - after_cleaning} rows with missing critical data")

# Clean and convert data types
if 'Date' in amazon_clean.columns:
    amazon_clean['Date'] = pd.to_datetime(amazon_clean['Date'], errors='coerce')
    amazon_clean = amazon_clean.dropna(subset=['Date'])

if 'Amount' in amazon_clean.columns:
    # Remove currency symbols and convert to numeric
    amazon_clean['Amount'] = amazon_clean['Amount'].astype(str).str.replace('₹', '').str.replace(',', '')
    amazon_clean['Amount'] = pd.to_numeric(amazon_clean['Amount'], errors='coerce')
    amazon_clean = amazon_clean.dropna(subset=['Amount'])

if 'Qty' in amazon_clean.columns:
    amazon_clean['Qty'] = pd.to_numeric(amazon_clean['Qty'], errors='coerce')
    amazon_clean = amazon_clean.dropna(subset=['Qty'])

print(f"Final cleaned Amazon dataset shape: {amazon_clean.shape}")

# Clean May 2022 Pricing Data
print("\n🔧 Cleaning May 2022 Pricing Data...")
may_2022_clean = may_2022.copy()

# Price columns to clean
price_columns = ['MRP Old Final', 'MRP Old', 'Ajio MRP', 'Amazon MRP', 'Amazon FBA MRP', 
                'Flipkart MRP', 'Limeroad MRP', 'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP']

for col in price_columns:
    if col in may_2022_clean.columns:
        may_2022_clean[col] = pd.to_numeric(may_2022_clean[col], errors='coerce')

print(f"May 2022 pricing data shape: {may_2022_clean.shape}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n⚙️ STEP 3: FEATURE ENGINEERING")
print("-" * 40)

# Feature engineering for Amazon sales data
if 'Date' in amazon_clean.columns:
    amazon_clean['Year'] = amazon_clean['Date'].dt.year
    amazon_clean['Month'] = amazon_clean['Date'].dt.month
    amazon_clean['Day'] = amazon_clean['Date'].dt.day
    amazon_clean['Weekday'] = amazon_clean['Date'].dt.day_name()
    amazon_clean['Quarter'] = amazon_clean['Date'].dt.quarter
    
    print("✅ Date-based features created")

# Revenue calculation
if 'Amount' in amazon_clean.columns and 'Qty' in amazon_clean.columns:
    amazon_clean['Revenue'] = amazon_clean['Amount'] * amazon_clean['Qty']
    amazon_clean['Unit_Price'] = amazon_clean['Amount'] / amazon_clean['Qty'].replace(0, 1)
    print("✅ Revenue and unit price features created")

# Category-based features
if 'Category' in amazon_clean.columns:
    amazon_clean['Category_Clean'] = amazon_clean['Category'].fillna('Unknown').str.strip().str.title()
    print("✅ Category features cleaned")

# Platform pricing comparison for May 2022 data
platform_columns = ['Ajio MRP', 'Amazon MRP', 'Amazon FBA MRP', 'Flipkart MRP', 
                    'Limeroad MRP', 'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP']

available_platforms = [col for col in platform_columns if col in may_2022_clean.columns]

if available_platforms:
    # Calculate average price across platforms
    may_2022_clean['Avg_Platform_Price'] = may_2022_clean[available_platforms].mean(axis=1, skipna=True)
    
    # Find minimum and maximum prices
    may_2022_clean['Min_Platform_Price'] = may_2022_clean[available_platforms].min(axis=1, skipna=True)
    may_2022_clean['Max_Platform_Price'] = may_2022_clean[available_platforms].max(axis=1, skipna=True)
    
    # Price spread
    may_2022_clean['Price_Spread'] = may_2022_clean['Max_Platform_Price'] - may_2022_clean['Min_Platform_Price']
    
    print("✅ Platform pricing features created")
# =============================================================================
# 3. EXPORT PROCESSED DATA (Xuất dữ liệu)
# =============================================================================
print("\n💾 Đang lưu file...")

# 1. Lưu file CSV
# index=False là RẤT QUAN TRỌNG: để nó không tự thêm một cột số thứ tự (0,1,2...) vào file
amazon_clean.to_csv('amazon_sales_cleaned.csv', index=False)
may_2022_clean.to_csv('pricing_data_cleaned.csv', index=False)

print("✅ Đã xuất file thành công!")
print(f"📂 File được lưu tại thư mục hiện tại: {os.getcwd()}") 

# Dòng trên giúp bạn biết chính xác file nằm ở đâu trong máy tính
