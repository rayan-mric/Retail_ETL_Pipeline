import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import calendar

DATA_PATH = "online_retail.csv"
DB_NAME = "Retail_Analytics.db"
TABLE_NAME = "sales"

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} : {message}")

def extract():
    log("Extracting raw data.")
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    return df

def transform(df):
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['CustomerID'] = df['CustomerID'].fillna(0).astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Monthly revenue
    monthly = df.groupby(df['InvoiceDate'].dt.to_period("M")).agg(
        Orders=('InvoiceNo', 'nunique'),
        Revenue=('Revenue', 'sum')
    ).reset_index()
    monthly['InvoiceDate'] = monthly['InvoiceDate'].astype(str)

    # Country revenue
    country = df.groupby('Country').agg(
        Orders=('InvoiceNo', 'nunique'),
        Revenue=('Revenue', 'sum')
    ).reset_index()

    # Top 10 products by revenue
    top_products = df.groupby('Description').agg(
        Orders=('InvoiceNo', 'nunique'),
        Revenue=('Revenue', 'sum')
    ).reset_index().sort_values('Revenue', ascending=False).head(10)

    customer = df.groupby(['CustomerID', 'Country']).agg(
        Orders=('InvoiceNo', 'nunique'),
        Revenue=('Revenue', 'sum')).reset_index()
    customer['Avg_Order_Value'] = np.round(customer['Revenue'] / customer['Orders'], 2)

    # Gold customers
    gold_customers = customer.groupby('Country').apply(lambda x: x.sort_values('Revenue', ascending=False).head(5)).reset_index(drop=True)
    gold_customers['Gold_Tier'] = 'Yes'
    repeat_customers = customer[customer['Orders'] > 1]
    repeat_rate = len(repeat_customers) / len(customer) * 100
    log(f"Customer Repeat Rate: {repeat_rate:.2f}%")
    return df, monthly, country, top_products, customer, gold_customers


def load(df, monthly, country, top_products, customer, gold_customers):
    conn = sqlite3.connect(DB_NAME)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    monthly.to_sql("agg_monthly_sales", conn, if_exists="replace", index=False)
    country.to_sql("agg_country_sales", conn, if_exists="replace", index=False)
    top_products.to_sql("top_products", conn, if_exists="replace", index=False)
    customer.to_sql("agg_customer_sales", conn, if_exists="replace", index=False)
    gold_customers.to_sql("gold_customers", conn, if_exists="replace", index=False)
    conn.close()
    log("All data loaded into SQLite database successfully.")

def visualize(monthly, top_products):
    months = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
    monthly['Month'] = pd.to_datetime(monthly['InvoiceDate']).dt.month.apply(lambda x: calendar.month_name[x])
    monthly_grouped = monthly.groupby('Month')['Revenue'].sum().reindex(months).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Monthly Revenue Trend
    axes[0].plot(monthly_grouped['Month'], monthly_grouped['Revenue'], marker='o', color='blue')
    axes[0].set_title('Monthly Revenue Trend')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Revenue ($)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True)

    # Top 10 Products
    axes[1].barh(top_products['Description'][::-1], top_products['Revenue'][::-1], color='skyblue')
    axes[1].set_title('Top 10 Products by Revenue')
    axes[1].set_xlabel('Revenue ($)')
    axes[1].set_ylabel('Product')

    plt.tight_layout()
    plt.show()


def main():
    log("ETL pipeline started.")

    raw_df = extract()
    df, monthly, country, top_products, customer, gold_customers = transform(raw_df)
    load(df, monthly, country, top_products, customer, gold_customers)
    visualize(monthly, top_products)

    log("ETL pipeline completed successfully.")

if __name__ == "__main__":
    main()
