# burn_classifier/local_analysis.py
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os


def calculate_dnbr(pre_fire_path, post_fire_path, output_dnbr_path):
    """
    Calculates dNBR from local TIF files using Rasterio.
    """
    with rasterio.open(pre_fire_path) as src_pre:
        pre_nbr = src_pre.read(1)
        profile = src_pre.profile  # Save metadata

    with rasterio.open(post_fire_path) as src_post:
        post_nbr = src_post.read(1)

    if pre_nbr.shape != post_nbr.shape:
        raise ValueError("Image size mismatch.")

    # Calculate dNBR (treat NaNs as 0)
    dnbr = np.nan_to_num(pre_nbr, nan=0.0) - np.nan_to_num(post_nbr, nan=0.0)

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    with rasterio.open(output_dnbr_path, 'w', **profile) as dst:
        dst.write(dnbr.astype(rasterio.float32), 1)

    print(f"dNBR image saved to: {output_dnbr_path}")
    return output_dnbr_path


def classify_severity(dnbr_tif_path, output_classified_path):
    """
    Classifies the dNBR TIF using Rasterio and Numpy.
    """
    with rasterio.open(dnbr_tif_path) as src:
        dnbr_array = src.read(1)
        profile = src.profile

    # Classes (1-7)
    conditions = [
        dnbr_array <= -0.251,
        (dnbr_array > -0.251) & (dnbr_array <= -0.101),
        (dnbr_array > -0.101) & (dnbr_array <= 0.100),
        (dnbr_array > 0.100) & (dnbr_array <= 0.270),
        (dnbr_array > 0.270) & (dnbr_array <= 0.440),
        (dnbr_array > 0.440) & (dnbr_array <= 0.660),
        dnbr_array > 0.660
    ]
    classes = [1, 2, 3, 4, 5, 6, 7]

    classified_array = np.select(conditions, classes, default=0).astype(rasterio.uint8)

    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    with rasterio.open(output_classified_path, 'w', **profile) as dst:
        dst.write(classified_array, 1)

    print(f"Classified image saved to: {output_classified_path}")
    return output_classified_path


def calculate_area(classified_tif_path):
    """
    Calculates the area for each severity class using Rasterio.
    """
    class_labels = {
        1: 'Enhanced Regrowth, High',
        2: 'Enhanced Regrowth, Moderate',
        3: 'Unburned',
        4: 'Low-Severity Burn',
        5: 'Moderate-Severity Burn',
        6: 'High-Severity Burn',
        7: 'Extreme-Severity Burn'
    }

    with rasterio.open(classified_tif_path) as src:
        classified_array = src.read(1)
        # Reproject the raster to an equal-distance projection (EPSG:3857)
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:3857', src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:3857',
            'transform': transform,
            'width': width,
            'height': height
        })

        reprojected_array = np.empty((height, width), dtype=classified_array.dtype)
        reproject(
            source=classified_array,
            destination=reprojected_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs='EPSG:3857',
            resampling=Resampling.nearest
        )

        pixel_size_x = transform.a  # meter
        pixel_size_y = abs(transform.e)  # meter
        pixel_area_m2 = pixel_size_x * pixel_size_y

    area_stats = {}
    total_area_m2 = 0

    unique_classes, counts = np.unique(classified_array, return_counts=True)

    for value, count in zip(unique_classes, counts):
        if value in class_labels:
            label = class_labels[value]
            area_m2 = count * pixel_area_m2
            area_stats[label] = {
                'class_id': int(value),
                'area_hectares': round(area_m2 / 10000, 2)
            }
            total_area_m2 += area_m2

    area_stats['Total_Analyzed_Area'] = {'area_hectares': round(total_area_m2 / 10000, 2)}

    print("Area calculation complete.")

    return area_stats


def plot_classified_map(classified_tif_path, title='Burn Severity Class'):
    """
    Visualizes the classified burn severity map.
    """
    class_labels = {
        1: 'Enhanced Regrowth, High',
        2: 'Enhanced Regrowth, Moderate',
        3: 'Unburned',
        4: 'Low-Severity Burn',
        5: 'Moderate-Severity Burn',
        6: 'High-Severity Burn',
        7: 'Extreme-Severity Burn'
    }
    # [cite: 9]
    severity_palette_hex = [
        '#1a9850', '#91cf60', '#d9ef8b', '#fee08b',
        '#fc8d59', '#d73027', '#7f0000'
    ]

    cmap = ListedColormap(severity_palette_hex)
    bounds = [1, 2, 3, 4, 5, 6, 7, 8]
    norm = BoundaryNorm(bounds, cmap.N)

    with rasterio.open(classified_tif_path) as src:
        data = src.read(1, masked=True)
        data = np.where(data == 0, np.nan, data)  # Mask out 0 values.
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use rasterio.plot.show to plot a raster
        show(data, ax=ax, cmap=cmap, norm=norm, transform=src.transform)

    # colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, ticks=[b + 0.5 for b in bounds[:-1]],
                        shrink=0.7)
    cbar.set_ticklabels([class_labels[i] for i in bounds[:-1]])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()