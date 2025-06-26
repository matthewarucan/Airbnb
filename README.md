![](/images/logo.png)

# 🏠📉 San Francisco Airbnb Regulation Impact Analysis

## 🧾 TL;DR

In January 2018, San Francisco implemented strict Airbnb regulations to reduce short-term rentals and improve housing availability. This project uses a **Difference-in-Differences (DiD)** model to assess whether those regulations stabilized **rent prices** and impacted **home values** in ZIP codes with high Airbnb activity.

**Key Takeaways:**

- 📉 **Rent growth slowed** in high-Airbnb ZIPs post-policy.
- 🏠 **Home values remained strong**, suggesting no negative impact on property investment.

---
## 📌 ASK Phase

### 🧠 Guiding Question  
What was the impact of **San Francisco’s 2018 Airbnb regulation**—specifically the *primary residence rule*—on **rent prices (ZORI)** and **home values (ZHVI)** across ZIP codes?

### 🏛️ Background on the Regulation  
In response to growing concerns about housing affordability, San Francisco introduced strict Airbnb regulations that went into full effect on **January 1, 2018**. Key elements of the regulation include:

- **Primary Residence Requirement**: Hosts are only allowed to rent out properties that are their *primary residence*.
- **90-Day Annual Cap**: Unhosted rentals (entire unit without the host present) are limited to **90 days per year**.
- **Mandatory Registration**: All hosts must register with the city to legally operate short-term rentals.
- **Platform Accountability**: Platforms like Airbnb must ensure all listings are registered and compliant with city regulations.

These rules aimed to reduce the number of full-time short-term rentals that contribute to housing shortages and drive up prices in the city.

### 🎯 Objective  
Apply a **Difference-in-Differences (DiD)** model to determine if ZIP codes with **high Airbnb activity** experienced significant changes in:

- **Rent Prices** (Zillow Observed Rent Index — ZORI)
- **Home Values** (Zillow Home Value Index — ZHVI)

---

## 🔄 PREPARE Phase

### 📁 Datasets Used

| Dataset Name               | Description                                            | Source               |
|---------------------------|--------------------------------------------------------|----------------------|
| `ZORI.csv`                | Rent prices by ZIP code                                | Zillow Research      |
| `ZHVI.csv`                | Home values by ZIP code                                | Zillow Research      |
| `listings.csv`            | Airbnb listings with location, price, and availability | Inside Airbnb        |
| `tl_2023_us_zcta520.shp`  | ZIP Code Tabulation Area shapefile for spatial join    | U.S. Census Bureau   |

### ⚙️ Data Preparation Steps

1. **Filter for ZIP-Level Data**  
   - Retain only rows where `RegionType == 'zip'` in ZORI/ZHVI datasets.

2. **Limit to Relevant Time Period**  
   - Keep monthly columns from **January 2016 to December 2020**.

3. **Clean Airbnb Listings**  
   - Remove rows with missing coordinates or extreme values.
   - Filter to active listings (`availability_365 > 30`) and `minimum_nights <= 30`.

4. **Assign ZIP Codes via Spatial Join**  
   - Convert listing coordinates to geometry points using `GeoPandas`.
   - Spatially join listings with San Francisco ZIP boundaries using a shapefile.

5. **Create Airbnb Intensity Groups**  
   - Count Airbnb listings per ZIP.
   - Group into quantiles: **Low**, **Medium**, and **High** Airbnb activity.

6. **Reshape ZORI and ZHVI Data**  
   - Transform from wide format (many monthly columns) to long format using `pd.melt`.

7. **Create DiD Variables**  
   - `treatment`: 1 if ZIP is in the **High Airbnb** group  
   - `post_policy`: 1 if date is after **January 1, 2018**  
   - `interaction`: `treatment * post_policy` — captures the causal effect

---
## 🧹 PROCESS Phase: Data Cleaning and Preprocessing

### Data Cleaning and Preprocessing
```python
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load & Clean ZORI and ZHVI
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

# Load & Clean Listings
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
```
---
## 📈 ANALYZE: Model and Evaluate

### 📊 Difference-in-Differences (DiD) Analysis
```python
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
```
---
## 🧹 SHARE Phase: Visualizations

### Rent Price Trends by Airbnb Group

```python
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
```
### Home Value Trends by Airbnb Group

```python
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
```

---
## 🎬 ACT Phase: Interpret & Conclude

### 🧾 Summary of Findings

The goal of this analysis was to evaluate the impact of San Francisco’s **Airbnb regulations**, implemented on **January 1, 2018**, on **rent prices** and **home values**, particularly in ZIP codes with high Airbnb activity.

---

### 💸 Rent Prices — Observed Effects

📉 **Rent prices in high-Airbnb ZIP codes remained relatively flat** after the 2018 policy, while **low-Airbnb areas saw continued growth until a COVID-era decline in 2020**.

**Interpretation**:  
The **Airbnb policy may have helped contain rent inflation** in high-activity areas by reducing short-term rental supply and keeping more housing stock available for long-term tenants.

---

### 🏠 Home Values — Observed Effects

🏠 **Home values increased across all ZIP groups**, including those with high Airbnb activity.

**Interpretation**:  
Despite Airbnb restrictions, **property values did not decline**, likely due to overall market trends or other macroeconomic factors. This suggests the policy didn’t have a suppressive effect on long-term investment demand in these neighborhoods.

---

### ✅ Conclusion

- **Rent prices** in high-Airbnb ZIPs were **stabilized post-policy**, suggesting the regulation was effective in **alleviating rental pressure**.
- **Home values** did **not suffer**, indicating the policy didn't deter long-term ownership interest.
- The **Difference-in-Differences (DiD)** approach provided evidence that **policy timing correlated with a shift in rent trends** in the most affected neighborhoods.

This analysis supports the idea that **targeted short-term rental regulations** can help stabilize rents **without harming long-term housing market confidence**.
