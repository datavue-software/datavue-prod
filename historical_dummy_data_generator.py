import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import json

np.random.seed(42) # Random at 42

def generate_customer_base():
    local_customers = [
        {"name": "Metro Wholesale Ltd", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "Yes", "discount_amount": 8},
        {"name": "City Bulk Foods", "category": "Local", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Region Foods Co", "category": "Local", "customer_company_size": "Small", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Prime Distributors", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
        {"name": "Local Grain Exchange", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Urban Bulk Supplies", "category": "Local", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "District Foods Inc", "category": "Local", "customer_company_size": "Mega", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Central Wholesale Co", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 3, "discount_offered": "Yes", "discount_amount": 5},
        {"name": "Town Grain Traders", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Municipal Food Supply", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Community Bulk Store", "category": "Local", "customer_company_size": "Small", "satisfaction_rating": 4, "discount_offered": "Yes", "discount_amount": 12},
        {"name": "Local Mart Chain", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "City Food Network", "category": "Local", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Regional Bulk Foods", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Metro Food Alliance", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
    ]
    local_customers += [
        {"name": "Harvest Depot Ltd", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 3, "discount_offered": "No", "discount_amount": 0},
        {"name": "Grain Barn Wholesale", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 2, "discount_offered": "No", "discount_amount": 0},
        {"name": "Sunrise Provisions", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Downtown Grains Ltd", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Hometown Bulk Market", "category": "Local", "customer_company_size": "Medium", "satisfaction_rating": 2, "discount_offered": "No", "discount_amount": 0},
        {"name": "Urban Provisions Inc", "category": "Local", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Central Foods Hub", "category": "Local", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Neighborhood Grains", "category": "Local", "customer_company_size": "Small", "satisfaction_rating": 2, "discount_offered": "No", "discount_amount": 0},
        {"name": "Compact Food Traders", "category": "Local", "customer_company_size": "Small", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Massive Grain Holdings", "category": "Local", "customer_company_size": "Mega", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
    ]

    international_customers = [
        {"name": "Global Grain Corp", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "Yes", "discount_amount": 7},
        {"name": "International Food Trade", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "World Maize Exchange", "category": "International", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Continental Supplies", "category": "International", "customer_company_size": "Mega", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
        {"name": "Ocean Foods International", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Cross Border Trading", "category": "International", "customer_company_size": "Mega", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Global Bulk Foods", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 3, "discount_offered": "No", "discount_amount": 0},
        {"name": "International Wholesale Co", "category": "International", "customer_company_size": "Small", "satisfaction_rating": 4, "discount_offered": "Yes", "discount_amount": 10},
        {"name": "World Food Network", "category": "International", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Maritime Traders Inc", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Export Trading Group", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Global Food Alliance", "category": "International", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "International Grain Co", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Overseas Food Supply", "category": "International", "customer_company_size": "Small", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "World Trade Foods", "category": "International", "customer_company_size": "Medium", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
    ]

    online_customers = [
        {"name": "E-Grain Trading", "category": "Online", "customer_company_size": "Small", "satisfaction_rating": 5, "discount_offered": "Yes", "discount_amount": 9},
        {"name": "Digital Food Exchange", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
        {"name": "Online Bulk Foods", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 3, "discount_offered": "No", "discount_amount": 0},
        {"name": "Virtual Trading Co", "category": "Online", "customer_company_size": "Large", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
        {"name": "E-Commerce Foods", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 3, "discount_offered": "No", "discount_amount": 0},
        {"name": "Digital Wholesale Network", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "Yes", "discount_amount": 6},
        {"name": "Cloud Trading Group", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
        {"name": "Online Mart Supply", "category": "Online", "customer_company_size": "Small", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Digital Food Alliance", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "E-Bulk Solutions", "category": "Online", "customer_company_size": "Large", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Virtual Food Trade", "category": "Online", "customer_company_size": "Mega", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Online Exchange Co", "category": "Online", "customer_company_size": "Mega", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "Digital Grain Store", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
        {"name": "E-Commerce Trades", "category": "Online", "customer_company_size": "Medium", "satisfaction_rating": 5, "discount_offered": "No", "discount_amount": 0},
        {"name": "Web Food Network", "category": "Online", "customer_company_size": "Mega", "satisfaction_rating": 4, "discount_offered": "No", "discount_amount": 0},
    ]

    customers_data = local_customers + international_customers + online_customers
    return pd.DataFrame(customers_data)

customers_df = generate_customer_base()
customers = customers_df.to_dict('records')
customers_to_leave = {}
customers_low_sales = [("International Grain Co"), ("Overseas Food Supply"), ("Massive Grain Holdings")] # We bring it down to 17% of total. 
decline_starts_at = 6 # 60% of dataset dates, and then it starts.

for customer in customers:

    customer_name = customer["name"];
    customer_satisfaction = customer["satisfaction_rating"];
    discount_offered = customer["discount_offered"]; 

    if customer_satisfaction < 4 and discount_offered == "No":

        customer_leaving = {**customer, "point_of_leave" : np.random.randint(6, 9) }
        customers_to_leave[customer_name] = customer_leaving
        
print(len(customers));
print(len(customers_to_leave));

for x in customers_to_leave:
    print(x)

warehouses = [
        {"warehouse_name": "Riverbend Warehouse", "warehouse_region": "North West"},
        {"warehouse_name": "Oakridge Storage", "warehouse_region": "North East"},
        {"warehouse_name": "Sunset Depot", "warehouse_region": "South West"},
        {"warehouse_name": "Pinehill Warehouse", "warehouse_region": "South East"},
        {"warehouse_name": "Maple Cross Dock", "warehouse_region": "Central"},
        {"warehouse_name": "Highland Storage", "warehouse_region": "Northern Highlands"},
        {"warehouse_name": "Seabreeze Warehouse", "warehouse_region": "East Coast"},
        {"warehouse_name": "Redrock Depot", "warehouse_region": "West Coast"},
        {"warehouse_name": "Midway Storage", "warehouse_region": "Midlands"},
        {"warehouse_name": "Greenfield Warehouse", "warehouse_region": "Southern Plains"},
    ]

products = [
        {"product_name": "White Maize", "price_per_ton": 210},
        {"product_name": "Yellow Maize", "price_per_ton": 200},
        {"product_name": "Organic Maize", "price_per_ton": 250}
        ]

# Create dates
n_records = 100
years = 1
total_days = 365 * years 
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=total_days)
dates = pd.date_range(start=start_date, end=end_date, periods=n_records)

# Helper to determine tons based on customer size
def generate_tons(customer_size, satisfaction_rating, discount_offered):
    if customer_size == 'Small':
        base = np.random.randint(1, 5) #  5, 20
    elif customer_size == 'Medium':
        base = np.random.randint(5, 20) # 20, 100
    elif customer_size == 'Large':
        base = np.random.randint(20, 100) # 100,300
    else:  # Mega
        base = np.random.randint(100, 250) # 300, 1000
    
    boost = 1.0
    if satisfaction_rating == 5:
        boost += 0.10  # +10%
    if discount_offered == "Yes":
        boost += 0.10  # +10%
    
    return int(base * boost)

def get_seasonal_factor(row):
    month = row['month']
    product = row['product_name']
    
    if product == 'White Maize':
        # Peak around Spring (March-May), centered at month 4
        seasonal = np.sin((month - 4) * np.pi / 6) * 0.2 + 1
    elif product == 'Yellow Maize':
        # Peak around Autumn (Sept-Nov), centered at month 9
        seasonal = np.sin((month - 9) * np.pi / 6) * 0.2 + 1
    elif product == 'Organic Maize':
        # Peak around Summer (June-Aug), centered at month 6
        seasonal = np.sin((month - 6) * np.pi / 6) * 0.2 + 1
    else:
        # Default (should not happen but safe fallback)
        seasonal = 1
    
    return seasonal


# Now build the data 
records = []

for date in dates:
    customer = np.random.choice(customers) # Gives a dataframe
    product = np.random.choice(products)
    warehouse = np.random.choice(warehouses)
    
    tons_sold = generate_tons(customer["customer_company_size"], customer["satisfaction_rating"], customer["discount_offered"])
    
    base_price_per_ton = product["price_per_ton"]
    discount = customer["discount_amount"] / 100 if customer["discount_offered"] == "Yes" else 0
    discount_price_per_ton = base_price_per_ton * (1 - discount)
    
    total_sale_amount = tons_sold * discount_price_per_ton
    
    record = {
        "date": date,
        "customer_name": customer["name"],
        "customer_category": customer["category"],
        "customer_company_size": customer["customer_company_size"],
        "satisfaction_rating": customer["satisfaction_rating"],
        "discount_offered": customer["discount_offered"],
        "discount_amount_percent": customer["discount_amount"],
        "product_name": product["product_name"],
        "base_price_per_ton": base_price_per_ton,
        "discount_price_per_ton": round(discount_price_per_ton, 2),
        "tons_sold": tons_sold,
        # "total_sale_amount": round(total_sale_amount, 2),
        "warehouse_name": warehouse["warehouse_name"],
        "warehouse_region": warehouse["warehouse_region"],
    }
    records.append(record)

df_ = pd.DataFrame(records)

df_[['month', 'year']] = df_['date'].apply(lambda df: pd.Series([df.month, df.year]))

# Create seasonal factor
df_["seasonal_factor"] = df_.apply(get_seasonal_factor, axis=1)

# Apply seasonal factor to tons_sold
df_['tons_sold_seasonal_amount'] = (df_['tons_sold'] * df_["seasonal_factor"]).astype(float) # round(0).astype(int)



df_["days_from_start"] = (df_['date'] - df_['date'].min()).dt.days
df_['years_from_start'] = df_["days_from_start"] / 365

def apply_growth_or_decline(row):
    years = row['years_from_start']
    size = row['customer_company_size']
    satisfaction = row['satisfaction_rating']
    discount_offered = row['discount_offered']
    
    if size == 'Small' and satisfaction < 4 and discount_offered == "No":
        # Steady Decline for bad small customers (up to -50% over 3 years)
        decline_rate = 0.15  # 15% per year
        factor = 1 - (years * decline_rate)
        factor = max(factor, 0)  # Do not go below 0
    else:
        # Gentle Growth for everyone else
        growth_rate = 0.05  # 5% per year
        factor = 1 + (years * growth_rate)
    
    return factor

def get_new_columns(row):
    
    customer_from_df = row["customer_name"]
    customer_from_dictionary = customers_to_leave.get(customer_from_df, 0)
    time_passed_perc = (row["days_from_start"] / total_days) * 10 # How mant days have passed


    if not customer_from_dictionary:
        return pd.Series({
        "days_from_start_10" : time_passed_perc,
        "point_of_leave": False,
        "do_not_deactivate" : True

    })

    customer_POL = customer_from_dictionary["point_of_leave"] # When they are to leave
    do_not_deactivate = True if float(customer_POL) <= time_passed_perc else False # Whether POL is less or equal to time_passed_perc 

    return pd.Series({
        "days_from_start_10" : time_passed_perc,
        "point_of_leave": customer_from_dictionary["point_of_leave"],
        "do_not_deactivate" : do_not_deactivate
    });


def assess_final_tonnes(row):

    tonnes_sold = row["tons_sold_seasonal_amount"]
    if not row["do_not_deactivate"]:
        return pd.Series({
            "final_tonnes_sold": row["tons_sold_seasonal_amount"] * 0.0,
    })
     
    return pd.Series({"final_tonnes_sold": row["tons_sold_seasonal_amount"]})


df_['growth_factor'] = df_.apply(apply_growth_or_decline, axis=1);
# df_[["point_of_leave", "normalized_days_from_start", "do_not_deactivate"]] = df_.apply(get_new_columns, axis=1)

df_[["days_from_start_10", "point_of_leave", "do_not_deactivate"]] = df_.apply(get_new_columns, axis=1);

df_["final_tons_sold"] = df_.apply(assess_final_tonnes, axis = 1)

df_["sale_amount"] = df_["tons_sold_seasonal_amount"] * df_["tons_sold_seasonal_amount"] * df_["discount_price_per_ton"]

df_ = df_.sort_values(["date"])

columns_to_remove = """
days_from_start years_from_start growth_factor days_from_start_10
point_of_leave do_not_deactivate month year seasonal_factor
discount_price_per_ton tons_sold
""".split()

df_.drop(columns=columns_to_remove, inplace=True)
df_.rename(columns={"date": "sale_date"}, inplace=True)
df_.drop(columns="tons_sold_seasonal_amount", inplace=True)
df_.reset_index(drop=True, inplace=True)
df_.to_csv("partial_csv.csv", index=False)

print(f"Run complete. Printed {n_records} rows"); 