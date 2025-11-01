# burn_classifier/__init__.py
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# import own library
from .gee_interface import display_aoi_map, fetch_nbr_images
from .local_analysis import calculate_dnbr, classify_severity, calculate_area, plot_classified_map


def auth_and_init():
    """Authenticate GEE and Google Drive."""
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        print("GEE authentication required...")
        ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    # Google Drive
    WORK_DIR = '/content/drive/MyDrive/GEE_exports'
    if not os.path.exists(WORK_DIR):
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
        except ImportError:
            os.makedirs(WORK_DIR, exist_ok=True)  # local

    print(f"Working directory set to: {WORK_DIR}")
    return WORK_DIR