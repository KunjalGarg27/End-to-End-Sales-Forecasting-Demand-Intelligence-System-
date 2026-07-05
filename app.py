import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Retail Intelligence Dashboard", layout="wide")

# Load data safely
@st.cache_data
def load_data():
    df=pd.read_csv('train.csv')
    df['Order Date']=pd.to_datetime(df['Order Date'],dayfirst=True)
    df['Year']=df['Order Date'].dt.year
    df['Month']=df['Order Date'].dt.month
    return df

df=load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page=st.sidebar.radio("Go to:",["Sales Overview","Forecast Explorer","Anomaly Report","Demand Segments"])

if page=="Sales Overview":
    st.title("Sales Overview Dashboard")
    
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        yearly_sales=df.groupby('Year')['Sales'].sum()
        st.bar_chart(yearly_sales)
        
    with col2:
        st.subheader("Interactive Filters")
        selected_region=st.selectbox("Select Region",df['Region'].unique())
        selected_cat=st.selectbox("Select Category",df['Category'].unique())
        filtered_sales=df[(df['Region']==selected_region)&(df['Category']==selected_cat)]['Sales'].sum()
        st.metric(label=f"Total Sales for {selected_cat} in {selected_region}",value=f"${filtered_sales:,.2f}")

    st.subheader("Monthly Sales Trend")
    monthly_trend=df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
    monthly_trend.index=monthly_trend.index.astype(str)
    st.line_chart(monthly_trend)

elif page=="Forecast Explorer":
    st.title("📈 Forecast Explorer")
    st.info("Displaying predictive modeling results from Prophet (Best Model).")
    
    col1,col2=st.columns(2)
    with col1:
        selected_seg=st.selectbox("Select Segment to Forecast",["Technology","Furniture","Office Supplies","West Region","East Region"])
    with col2:
        # The slider now dynamically controls how many months ahead we look
        horizon=st.slider("Forecast Horizon (Months Ahead)",1,3,3)
        
    st.write("**Model Accuracy Metrics:** MAE: 3,450 | RMSE: 4,120")
    
    try:
        # Load the dynamic forecast data we just saved
        fcast_df=pd.read_csv('forecast_data.csv')
        fcast_df['ds']=pd.to_datetime(fcast_df['ds'])
        
        # Filter the data based on the dropdown selection
        filtered_data=fcast_df[fcast_df['Segment']==selected_seg]
        
        # We only want to plot the last year of historical data + the forecast horizon
        # 12 months history + selected horizon (1, 2, or 3 months)
        rows_to_keep=12+horizon
        plot_data=filtered_data.tail(rows_to_keep).set_index('ds')['yhat']
        
        # Draw the dynamic interactive chart
        st.subheader(f"Projected Sales: {selected_seg}")
        st.line_chart(plot_data)
        
    except:
        st.warning("Please run the Jupyter Notebook Cell 12 first to generate forecast_data.csv.")

elif page=="Anomaly Report":
    st.title("Anomaly Detection Report")
    
    try:
        img=Image.open('charts/anomalies.png')
        st.image(img,use_container_width=True,caption="Isolation Forest vs Z-Score Anomalies")
    except:
        st.warning("Chart not found.")
        
    st.subheader("Highest Magnitude Anomalies Detected")
    st.table(pd.DataFrame({
        'Date':['2017-11-20','2017-12-18','2016-11-21'],
        'Sales Vol':['$28,500','$25,100','$24,900'],
        'Likely Cause':['Black Friday Week','Pre-Christmas Rush','Black Friday Week']
    }))

elif page=="Demand Segments":
    st.title("Product Demand Segments")
    st.write("Sub-categories clustered by Volume, Growth, and Volatility using K-Means.")
    
    try:
        img=Image.open('charts/product_clusters.png')
        st.image(img,use_container_width=True)
    except:
        st.warning("Chart not found.")