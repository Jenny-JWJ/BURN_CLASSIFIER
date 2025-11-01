# burn_classifier/gee_interface.py
import ee
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import geopy.geocoders
import contextily as ctx
from shapely.geometry import Polygon

# Set the global user_agent to comply with Nominatim's policy
geopy.geocoders.options.default_user_agent = "burn_analysis_project_v1"


# Masks clouds and cirrus in S2 images
def _maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)


def display_aoi_map(aoi, location_name="Area of Interest", zoom_level=12):
    """
    Displays a map of a known ee.Geometry object with a basemap.

    Parameters:
    - aoi (ee.Geometry): The Earth Engine Geometry object (e.g., ee.Geometry.Rectangle).
    - location_name (str): A name for the map title.
    - zoom_level (int): Basemap zoom level (larger number = closer zoom).
    """
    print(f"Displaying map for: {location_name}...")

    try:
        coords = aoi.getInfo()['coordinates']
    except Exception as e:
        print(f"Error getting coordinates from GEE: {e}")
        print("Please ensure you have authenticated and initialized Earth Engine.")
        return

    # 2. Convert GEE coordinates to a Shapely Polygon
    # For a simple rectangle, the coordinates are stored in coords[0].
    if not coords or not coords[0]:
        raise ValueError("AOI geometry is empty or invalid.")

    shapely_poly = Polygon(coords[0])

    # 3. Create a GeoDataFrame
    # GEE uses WGS84 (EPSG:4326) by default.
    gdf = gpd.GeoDataFrame([1], geometry=[shapely_poly], crs="EPSG:4326")

    # 4. Convert to Web Mercator (EPSG:3857) for use with contextily
    gdf_wm = gdf.to_crs(epsg=3857)

    # 5. Plot the map
    ax = gdf_wm.plot(figsize=(10, 5), edgecolor='red', facecolor='none', linewidth=2)

    try:
        # use contextily to add basemap
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=zoom_level)
    except Exception as e:
        print(f"Could not add basemap (may require 'pip install contextily'). Error: {e}")

    ax.set_title(f"Area of Interest: {location_name}")
    ax.set_axis_off()
    plt.show()

    return ax

def fetch_nbr_images(aoi, pre_dates, post_dates, folder, aoi_name="fire_event", scale=10):
    """
    Calculates pre- and post-fire NBR images and starts export tasks to Google Drive.
    """
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        print("Please authenticate GEE first (ee.Authenticate()).")
        raise e

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)

    # Pre-fire image
    pre_image = s2.filterDate(pre_dates[0], pre_dates[1]).map(_maskS2clouds).median().clip(aoi)
    pre_nbr = pre_image.normalizedDifference(['B8', 'B12']).rename('pre_nbr')

    # Post-fire image
    post_image = s2.filterDate(post_dates[0], post_dates[1]).map(_maskS2clouds).median().clip(aoi)
    post_nbr = post_image.normalizedDifference(['B8', 'B12']).rename('post_nbr')

    projection = pre_nbr.projection().crs()

    # Start export tasks
    pre_filename = f"{aoi_name}_pre_nbr"
    post_filename = f"{aoi_name}_post_nbr"

    # export
    for img, name in [(pre_nbr, pre_filename), (post_nbr, post_filename)]:
        print(f"Starting GEE task: {name}...")
        task = ee.batch.Export.image.toDrive(
            image=img.toFloat(),
            description=name,
            folder=folder,
            fileNamePrefix=name,
            region=aoi,
            scale=scale,
            crs=projection,
            maxPixels=1e13
        )
        task.start()

    print("\n---")
    print(f"Tasks started. Please check the GEE 'Tasks' tab and proceed after completion.")
    print(f"Files will be exported to your Google Drive '{folder}' folder.")

    return f"{pre_filename}.tif", f"{post_filename}.tif"