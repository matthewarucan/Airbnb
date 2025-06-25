## 🧹 PHASE 1: Data Cleaning and Preprocessing
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
