import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS tambahan
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample data ecommerce sebagai fallback
def create_sample_data():
    """Membuat sample data jika file tidak ditemukan"""
    np.random.seed(42)
    n_records = 1000
    
    sample_data = {
        'InvoiceNo': np.random.choice([f'55{np.random.randint(1000,9999)}' for _ in range(100)], n_records),
        'StockCode': np.random.choice(['85099B', '47566', '22423', '85123A', '22197', '84991'], n_records),
        'Description': np.random.choice([
            'JUMBO BAG RED RETROSPOT', 'PARTY BUNTING', 'REGENCY CAKESTAND 3 TIER',
            'WHITE HANGING HEART T-LIGHT HOLDER', '60 TEATIME FAIRY CAKE CASES'
        ], n_records),
        'Quantity': np.random.randint(1, 50, n_records),
        'InvoiceDate': pd.date_range('2021-01-01', '2021-12-31', n_records),
        'UnitPrice': np.round(np.random.uniform(0.5, 10, n_records), 2),
        'CustomerID': np.random.randint(12000, 18000, n_records),
        'Country': np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Italy'], n_records)
    }
    
    return pd.DataFrame(sample_data)

def load_data():
    """Load data dari file ecommerce.csv atau gunakan sample data"""
    try:
        # Coba load dari file
        df = pd.read_csv('ecommerce.csv')
        st.sidebar.success("✅ Data loaded from ecommerce.csv")
        
        # Konversi InvoiceDate ke datetime
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
        except:
            try:
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            except:
                st.sidebar.warning("⚠️ Could not convert InvoiceDate to datetime")
        
        return df
    except FileNotFoundError:
        st.sidebar.warning("⚠️ ecommerce.csv not found. Using sample data.")
        return create_sample_data()
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}. Using sample data.")
        return create_sample_data()

class ECommerceAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.preprocess_data()
    
    def preprocess_data(self):
        """Preprocessing data untuk analisis"""
        # Pastikan InvoiceDate sudah datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['InvoiceDate']):
            try:
                self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'], format='%m/%d/%Y %H:%M')
            except:
                try:
                    self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
                except:
                    st.error("Error converting InvoiceDate to datetime")
                    return
        
        # Ekstrak komponen tanggal
        self.data['Year'] = self.data['InvoiceDate'].dt.year
        self.data['Month'] = self.data['InvoiceDate'].dt.month
        self.data['Day'] = self.data['InvoiceDate'].dt.day
        self.data['Hour'] = self.data['InvoiceDate'].dt.hour
        self.data['DayOfWeek'] = self.data['InvoiceDate'].dt.day_name()
        
        # Hitung total sales
        self.data['TotalSales'] = self.data['Quantity'] * self.data['UnitPrice']
        
        # Kategorikan waktu
        self.data['TimeOfDay'] = pd.cut(self.data['Hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)

def get_date_range_info(df):
    """Mendapatkan informasi date range dengan handling error"""
    if 'InvoiceDate' not in df.columns:
        return "N/A", "N/A"
    
    try:
        if pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
            min_date = df['InvoiceDate'].min().strftime('%Y-%m-%d')
            max_date = df['InvoiceDate'].max().strftime('%Y-%m-%d')
        else:
            # Jika masih string, coba konversi dulu
            temp_dates = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            min_date = temp_dates.min().strftime('%Y-%m-%d') if not pd.isna(temp_dates.min()) else "N/A"
            max_date = temp_dates.max().strftime('%Y-%m-%d') if not pd.isna(temp_dates.max()) else "N/A"
        
        return min_date, max_date
    except:
        return "N/A", "N/A"

def main():
    # Header utama
    st.markdown('<h1 class="main-header">🛒 E-Commerce Analytics Portfolio</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Exploratory Data Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar untuk navigasi dan info dataset
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"📊 Records: {len(df):,}")
    
    # Get date range info dengan error handling
    min_date, max_date = get_date_range_info(df)
    st.sidebar.write(f"📅 Date Range: {min_date} to {max_date}")
    
    st.sidebar.write(f"🌍 Countries: {df['Country'].nunique() if 'Country' in df.columns else 'N/A'}")
    
    # Hapus Customer Analysis dari opsi
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["📊 Overview", "📈 Sales Analysis", "🌍 Geographic Analysis", 
         "📦 Product Analysis", "🕒 Time Analysis", "📥 Data Export"]
    )
    
    # Inisialisasi analyzer
    analyzer = ECommerceAnalyzer(df)
    data = analyzer.data
    
    # Tampilkan berdasarkan pilihan
    if analysis_type == "📊 Overview":
        show_overview(data, analyzer)
    elif analysis_type == "📈 Sales Analysis":
        show_sales_analysis(data)
    elif analysis_type == "🌍 Geographic Analysis":
        show_geographic_analysis(data)
    elif analysis_type == "📦 Product Analysis":
        show_product_analysis(data)
    elif analysis_type == "🕒 Time Analysis":
        show_time_analysis(data)
    elif analysis_type == "📥 Data Export":
        show_data_export(data)

def show_overview(data, analyzer):
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = data['TotalSales'].sum()
        st.metric("Total Revenue", f"£{total_sales:,.2f}")
    
    with col2:
        total_transactions = data['InvoiceNo'].nunique()
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col3:
        total_customers = data['CustomerID'].nunique()
        st.metric("Unique Customers", f"{total_customers:,}")
    
    with col4:
        total_products = data['StockCode'].nunique()
        st.metric("Unique Products", f"{total_products:,}")
    
    # Dataset preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Statistics")
        numerical_cols = ['Quantity', 'UnitPrice', 'TotalSales']
        available_numerical = [col for col in numerical_cols if col in data.columns]
        if available_numerical:
            st.dataframe(data[available_numerical].describe(), use_container_width=True)
        else:
            st.write("No numerical columns available")
    
    with col2:
        st.subheader("Categorical Summary")
        categorical_cols = ['Country', 'StockCode', 'Description']
        categorical_data = []
        for col in categorical_cols:
            if col in data.columns:
                categorical_data.append({
                    'Column': col,
                    'Unique Values': data[col].nunique()
                })
        if categorical_data:
            st.dataframe(pd.DataFrame(categorical_data), use_container_width=True)
        else:
            st.write("No categorical columns available")
    
    # Data quality check
    st.subheader("Data Quality Check")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_data = data.isnull().sum().sum()
        st.info(f"Missing Values: {missing_data}")
    
    with col2:
        duplicate_rows = data.duplicated().sum()
        st.info(f"Duplicate Rows: {duplicate_rows}")
    
    with col3:
        memory_usage = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        st.info(f"Memory Usage: {memory_usage}")

def show_sales_analysis(data):
    st.markdown('<h2 class="section-header">📈 Sales Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        if pd.api.types.is_datetime64_any_dtype(data['InvoiceDate']):
            min_date = data['InvoiceDate'].min().date()
            max_date = data['InvoiceDate'].max().date()
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            st.warning("InvoiceDate is not in datetime format")
            date_range = (data['InvoiceDate'].min(), data['InvoiceDate'].max())
    
    with col2:
        selected_countries = st.multiselect(
            "Select Countries",
            options=data['Country'].unique() if 'Country' in data.columns else [],
            default=data['Country'].unique()[:3] if 'Country' in data.columns else []
        )
    
    # Filter data
    if pd.api.types.is_datetime64_any_dtype(data['InvoiceDate']):
        if len(selected_countries) > 0 and 'Country' in data.columns:
            filtered_data = data[
                (data['InvoiceDate'].dt.date >= date_range[0]) & 
                (data['InvoiceDate'].dt.date <= date_range[1]) &
                (data['Country'].isin(selected_countries))
            ]
        else:
            filtered_data = data[
                (data['InvoiceDate'].dt.date >= date_range[0]) & 
                (data['InvoiceDate'].dt.date <= date_range[1])
            ]
    else:
        filtered_data = data
        st.warning("Cannot filter by date - InvoiceDate not in datetime format")
    
    # Sales metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not filtered_data.empty:
            avg_transaction_value = filtered_data.groupby('InvoiceNo')['TotalSales'].sum().mean()
            st.metric("Avg Transaction Value", f"£{avg_transaction_value:.2f}")
        else:
            st.metric("Avg Transaction Value", "N/A")
    
    with col2:
        if not filtered_data.empty:
            avg_quantity = filtered_data['Quantity'].mean()
            st.metric("Avg Quantity per Item", f"{avg_quantity:.1f}")
        else:
            st.metric("Avg Quantity per Item", "N/A")
    
    with col3:
        if not filtered_data.empty and 'Description' in filtered_data.columns:
            top_product = filtered_data.groupby('Description')['TotalSales'].sum()
            if not top_product.empty:
                top_product_name = top_product.idxmax()
                st.metric("Top Product by Revenue", top_product_name[:20] + "...")
            else:
                st.metric("Top Product by Revenue", "N/A")
        else:
            st.metric("Top Product by Revenue", "N/A")
    
    with col4:
        if not filtered_data.empty and 'Hour' in filtered_data.columns:
            peak_hour = filtered_data['Hour'].mode()
            if not peak_hour.empty:
                st.metric("Peak Sales Hour", f"{peak_hour.iloc[0]}:00")
            else:
                st.metric("Peak Sales Hour", "N/A")
        else:
            st.metric("Peak Sales Hour", "N/A")
    
    # Sales trends
    st.subheader("Sales Trends Over Time")
    
    if not filtered_data.empty and pd.api.types.is_datetime64_any_dtype(filtered_data['InvoiceDate']):
        # Daily sales
        daily_sales = filtered_data.groupby(filtered_data['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
        fig_daily = px.line(daily_sales, x='InvoiceDate', y='TotalSales', 
                           title='Daily Sales Trend')
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Monthly sales by country
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_sales = filtered_data.groupby([filtered_data['InvoiceDate'].dt.month, 'Country'])['TotalSales'].sum().reset_index()
            if not monthly_sales.empty:
                fig_monthly = px.bar(monthly_sales, x='InvoiceDate', y='TotalSales', color='Country',
                                   title='Monthly Sales by Country')
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Sales distribution
            transaction_sales = filtered_data.groupby('InvoiceNo')['TotalSales'].sum()
            fig_dist = px.box(x=transaction_sales, title='Sales Distribution per Transaction')
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning("No time-based data available for analysis")

def show_geographic_analysis(data):
    st.markdown('<h2 class="section-header">🌍 Geographic Analysis</h2>', unsafe_allow_html=True)
    
    if 'Country' not in data.columns:
        st.warning("Country data not available")
        return
    
    # Country-wise metrics
    country_stats = data.groupby('Country').agg({
        'TotalSales': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
    
    country_stats.columns = ['Country', 'Total Revenue', 'Number of Transactions', 'Unique Customers', 'Total Quantity']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Countries by Revenue")
        top_countries = country_stats.nlargest(10, 'Total Revenue')[['Country', 'Total Revenue']]
        fig_country = px.bar(top_countries, x='Total Revenue', y='Country', 
                           orientation='h', title='Top 10 Countries by Revenue')
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        st.subheader("Revenue Distribution by Country")
        fig_pie = px.pie(country_stats, values='Total Revenue', names='Country',
                        title='Revenue Distribution by Country')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed country table
    st.subheader("Detailed Country Statistics")
    st.dataframe(country_stats.sort_values('Total Revenue', ascending=False), use_container_width=True)
    
    # Geographic patterns
    st.subheader("Customer Distribution by Country")
    customer_dist = data.groupby('Country')['CustomerID'].nunique().reset_index()
    customer_dist = customer_dist.sort_values('CustomerID', ascending=False)
    
    fig_customers = px.treemap(customer_dist, path=['Country'], values='CustomerID',
                             title='Customer Distribution by Country')
    st.plotly_chart(fig_customers, use_container_width=True)

def show_product_analysis(data):
    st.markdown('<h2 class="section-header">📦 Product Performance Analysis</h2>', unsafe_allow_html=True)
    
    if 'StockCode' not in data.columns or 'Description' not in data.columns:
        st.warning("Product data not available")
        return
    
    # Product statistics
    product_stats = data.groupby(['StockCode', 'Description']).agg({
        'TotalSales': 'sum',
        'Quantity': 'sum',
        'InvoiceNo': 'nunique',
        'UnitPrice': 'mean'
    }).reset_index()
    
    product_stats.columns = ['StockCode', 'Description', 'TotalRevenue', 'TotalQuantity', 'TransactionCount', 'AvgPrice']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Products by Revenue")
        top_products = product_stats.nlargest(10, 'TotalRevenue')[['Description', 'TotalRevenue', 'TotalQuantity']]
        fig_products = px.bar(top_products, x='TotalRevenue', y='Description',
                            orientation='h', title='Top 10 Products by Revenue')
        st.plotly_chart(fig_products, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Quantity Analysis")
        fig_scatter = px.scatter(product_stats, x='AvgPrice', y='TotalQuantity',
                               size='TotalRevenue', color='TotalRevenue',
                               hover_data=['Description'],
                               title='Price vs Quantity Sold')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Product categories analysis
    st.subheader("Product Performance Metrics")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not product_stats.empty:
            best_seller = product_stats.loc[product_stats['TotalQuantity'].idxmax(), 'Description']
            st.metric("Best Selling Product", best_seller[:20] + "...")
        else:
            st.metric("Best Selling Product", "N/A")
    
    with col2:
        if not product_stats.empty:
            highest_revenue = product_stats.loc[product_stats['TotalRevenue'].idxmax(), 'Description']
            st.metric("Highest Revenue Product", highest_revenue[:20] + "...")
        else:
            st.metric("Highest Revenue Product", "N/A")
    
    with col3:
        avg_product_price = product_stats['AvgPrice'].mean()
        st.metric("Average Product Price", f"£{avg_product_price:.2f}")
    
    with col4:
        unique_products = len(product_stats)
        st.metric("Unique Products", f"{unique_products}")
    
    # Product table
    st.subheader("Product Performance Details")
    st.dataframe(product_stats.sort_values('TotalRevenue', ascending=False).head(20), use_container_width=True)

def show_time_analysis(data):
    st.markdown('<h2 class="section-header">🕒 Temporal Analysis</h2>', unsafe_allow_html=True)
    
    if not pd.api.types.is_datetime64_any_dtype(data['InvoiceDate']):
        st.warning("InvoiceDate is not in datetime format - cannot perform time analysis")
        return
    
    # Time-based aggregations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by hour of day
        if 'Hour' in data.columns:
            hourly_sales = data.groupby('Hour')['TotalSales'].sum().reset_index()
            fig_hourly = px.line(hourly_sales, x='Hour', y='TotalSales',
                               title='Sales by Hour of Day')
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Sales by day of week
        if 'DayOfWeek' in data.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_sales = data.groupby('DayOfWeek')['TotalSales'].sum().reindex(day_order).reset_index()
            fig_daily = px.bar(daily_sales, x='DayOfWeek', y='TotalSales',
                             title='Sales by Day of Week')
            st.plotly_chart(fig_daily, use_container_width=True)
    
    # Monthly trends
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_trend = data.groupby(data['InvoiceDate'].dt.month)['TotalSales'].sum().reset_index()
        fig_monthly = px.line(monthly_trend, x='InvoiceDate', y='TotalSales',
                            title='Monthly Sales Trend')
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Time of day analysis
        if 'TimeOfDay' in data.columns:
            time_sales = data.groupby('TimeOfDay')['TotalSales'].sum().reset_index()
            fig_time = px.pie(time_sales, values='TotalSales', names='TimeOfDay',
                            title='Sales Distribution by Time of Day')
            st.plotly_chart(fig_time, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    
    # Create seasonal categories
    data['Season'] = data['InvoiceDate'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    seasonal_sales = data.groupby('Season')['TotalSales'].sum().reset_index()
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_sales['Season'] = pd.Categorical(seasonal_sales['Season'], categories=season_order, ordered=True)
    seasonal_sales = seasonal_sales.sort_values('Season')
    
    fig_seasonal = px.bar(seasonal_sales, x='Season', y='TotalSales',
                         title='Seasonal Sales Performance')
    st.plotly_chart(fig_seasonal, use_container_width=True)

def show_data_export(data):
    st.markdown('<h2 class="section-header">📥 Data Export</h2>', unsafe_allow_html=True)
    
    st.subheader("Processed Data Download")
    
    # Pilih format download
    export_format = st.radio("Select Export Format:", ["CSV", "Excel", "JSON"])
    
    # Nama file
    filename = st.text_input("Filename:", "ecommerce_processed_data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Processed Data"):
            if export_format == "CSV":
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data.to_excel(writer, index=False, sheet_name='Processed_Data')
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:  # JSON
                json_str = data.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("📊 Download Summary Report"):
            # Buat summary report
            summary_data = {
                'Metric': ['Total Records', 'Total Revenue', 'Unique Customers', 'Unique Products', 'Date Range'],
                'Value': [
                    len(data),
                    f"£{data['TotalSales'].sum():,.2f}",
                    data['CustomerID'].nunique() if 'CustomerID' in data.columns else 'N/A',
                    data['StockCode'].nunique() if 'StockCode' in data.columns else 'N/A',
                    f"{data['InvoiceDate'].min().strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(data['InvoiceDate']) else 'N/A'} to {data['InvoiceDate'].max().strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(data['InvoiceDate']) else 'N/A'}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
    
    with col3:
        if st.button("🔄 Reset Filters"):
            st.experimental_rerun()
    
    # Tampilkan processed data
    st.subheader("Processed Data Preview")
    st.dataframe(data.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
