# ----------------------------------------
# 🧹 PHASE 1: Data Cleaning and Preprocessing
# ----------------------------------------

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# ---------------- Load & Clean ZORI and ZHVI ----------------
zori_raw = pd.read_csv("/Users/matthewarucan/Desktop/ZORI.csv")
zhvi_raw = pd.read_csv("/Users/matthewarucan/Desktop/ZHVI.csv")

# Keep only ZIP-level rows
zori_zip = zori_raw[zori_raw['RegionType'] == 'zip'].copy()
zhvi_zip = zhvi_raw[zhvi_raw['RegionType'] == 'zip'].copy()

# Identify date columns in desired range
date_columns = [col for col in zori_zip.columns if '2016-01-31' <= col <= '2020-12-31']

# Keep ZIP code and relevant dates
zori_clean = zori_zip[['RegionName'] + date_columns].copy().rename(columns={'RegionName': 'zipcode'})
zhvi_clean = zhvi_zip[['RegionName'] + date_columns].copy().rename(columns={'RegionName': 'zipcode'})

# Save cleaned data
zori_clean.to_csv("zori_clean_2016_2020.csv", index=False)
zhvi_clean.to_csv("zhvi_clean_2016_2020.csv", index=False)
print("✅ Saved ZORI and ZHVI cleaned datasets.")

# ---------------- Load & Clean Listings ----------------
listings = pd.read_csv("/Users/matthewarucan/Desktop/listings.csv")

# Filter relevant columns and drop invalid listings
listings = listings[['id', 'latitude', 'longitude', 'availability_365', 'price', 'minimum_nights']]
listings = listings.dropna(subset=['latitude', 'longitude'])
listings = listings[listings['availability_365'] > 30]
listings = listings[listings['minimum_nights'] <= 30]

# Create geometry and convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(listings['longitude'], listings['latitude'])]
listings_gdf = gpd.GeoDataFrame(listings, geometry=geometry, crs="EPSG:4326")

# Load ZIP shapefile and filter to San Francisco ZIPs
zcta = gpd.read_file("/Users/matthewarucan/Desktop/tl_2023_us_zcta520/tl_2023_us_zcta520.shp")
zcta_sf = zcta[zcta['ZCTA5CE20'].str.startswith('941')].copy().to_crs("EPSG:4326")

# Spatial join listings with ZIP shapes
listings_with_zip = gpd.sjoin(listings_gdf, zcta_sf, how="left", predicate='within')
listings_with_zip = listings_with_zip.rename(columns={'ZCTA5CE20': 'zipcode'})
listings_with_zip = listings_with_zip[['id', 'latitude', 'longitude', 'zipcode', 'availability_365', 'price', 'minimum_nights']]
listings_with_zip.to_csv("listings_with_zip.csv", index=False)
print("✅ Saved listings with ZIP codes.")

# ----------------------------------------
# 📊 PHASE 2:: DiD Analysis
# ----------------------------------------

# Step 1: Count listings per ZIP and group by intensity
zip_counts = listings_with_zip['zipcode'].value_counts().reset_index()
zip_counts.columns = ['zipcode', 'airbnb_listings']
zip_counts['airbnb_group'] = pd.qcut(zip_counts['airbnb_listings'], q=3, labels=['Low', 'Medium', 'High'])


# Step 2: Convert ZIP codes to string for join compatibility
zori_clean['zipcode'] = zori_clean['zipcode'].astype(str)
zhvi_clean['zipcode'] = zhvi_clean['zipcode'].astype(str)
zip_counts['zipcode'] = zip_counts['zipcode'].astype(str)

# Step 3: Merge Airbnb groups into rent and home value data
zori_tagged = pd.merge(zori_clean, zip_counts[['zipcode', 'airbnb_group']], on='zipcode', how='left')
zhvi_tagged = pd.merge(zhvi_clean, zip_counts[['zipcode', 'airbnb_group']], on='zipcode', how='left')

# Step 4: Reshape to long format
zori_long = zori_tagged.melt(id_vars=['zipcode', 'airbnb_group'], var_name='date', value_name='rent_price')
zhvi_long = zhvi_tagged.melt(id_vars=['zipcode', 'airbnb_group'], var_name='date', value_name='home_value')

# Step 5: Create DiD variables
zori_long['date'] = pd.to_datetime(zori_long['date'])
zhvi_long['date'] = pd.to_datetime(zhvi_long['date'])

policy_date = pd.to_datetime('2018-01-01')
zori_long['post_policy'] = (zori_long['date'] >= policy_date).astype(int)
zhvi_long['post_policy'] = (zhvi_long['date'] >= policy_date).astype(int)
zori_long['treatment'] = (zori_long['airbnb_group'] == 'High').astype(int)
zhvi_long['treatment'] = (zhvi_long['airbnb_group'] == 'High').astype(int)
zori_long['interaction'] = zori_long['treatment'] * zori_long['post_policy']
zhvi_long['interaction'] = zhvi_long['treatment'] * zhvi_long['post_policy']

import statsmodels.formula.api as smf

# Step 6: Run DiD models
zori_model = smf.ols("rent_price ~ treatment + post_policy + interaction", data=zori_long.dropna()).fit()
zhvi_model = smf.ols("home_value ~ treatment + post_policy + interaction", data=zhvi_long.dropna()).fit()

print("\n=== DID RESULTS: Rent Prices ===")
print(zori_model.summary())

print("\n=== DID RESULTS: Home Values ===")
print(zhvi_model.summary())

# ----------------------------------------
# 📈 PHASE 3: Visualizations
# ----------------------------------------

# Rent Price Trends by Airbnb Group
zori_plot = zori_long.groupby(['date', 'airbnb_group'])['rent_price'].mean().reset_index()
plt.figure(figsize=(10, 6))
for group in zori_plot['airbnb_group'].unique():
    data = zori_plot[zori_plot['airbnb_group'] == group]
    plt.plot(data['date'], data['rent_price'], label=f"{group} Airbnb ZIPs")
plt.axvline(policy_date, color='red', linestyle='--', label='Policy Start')
plt.title('Rent Price Trends by Airbnb Intensity')
plt.xlabel('Date')
plt.ylabel('Average Rent Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Home Value Trends by Airbnb Group
zhvi_plot = zhvi_long.groupby(['date', 'airbnb_group'])['home_value'].mean().reset_index()
plt.figure(figsize=(10, 6))
for group in zhvi_plot['airbnb_group'].unique():
    data = zhvi_plot[zhvi_plot['airbnb_group'] == group]
    plt.plot(data['date'], data['home_value'], label=f"{group} Airbnb ZIPs")
plt.axvline(policy_date, color='red', linestyle='--', label='Policy Start')
plt.title('Home Value Trends by Airbnb Intensity')
plt.xlabel('Date')
plt.ylabel('Average Home Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
