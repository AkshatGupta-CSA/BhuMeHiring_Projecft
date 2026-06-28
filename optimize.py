#!/usr/bin/env python3
"""
Localized boundary optimizer for BhuMe plots.

Aligns plot boundaries by matching them against local binary edge maps in boundaries.tif
using Euclidean Distance Transform (EDT) shape matching and a two-stage grid search.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from scipy.ndimage import distance_transform_edt, map_coordinates
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon

from bhume import load, score, write_predictions
from bhume.baseline import global_median_shift

DEFAULT_VILLAGE = 'data/34855_vadnerbhairav_chandavad_nashik'

def get_boundary_points(geom, step_m=2.0):
    """Sample points along the exterior boundary of a geometry."""
    points = []
    if isinstance(geom, Polygon):
        ext = geom.exterior
        length = ext.length
        dists = np.arange(0, length, step_m)
        for d in dists:
            pt = ext.interpolate(d)
            points.append((pt.x, pt.y))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            ext = poly.exterior
            length = ext.length
            dists = np.arange(0, length, step_m)
            for d in dists:
                pt = ext.interpolate(d)
                points.append((pt.x, pt.y))
    return np.array(points)

def main(village_dir: str) -> None:
    village = load(village_dir)
    print(f"Loaded village: {village.slug}")
    
    # 1. Fallback to global median shift baseline if boundaries.tif is missing
    if not village.boundaries_path or not Path(village.boundaries_path).exists():
        print("Warning: boundaries.tif not found. Falling back to baseline global median shift.")
        preds = global_median_shift(village)
        out = write_predictions(Path(village_dir) / 'predictions.geojson', preds)
        print(f"Wrote predictions to {out}")
        if village.example_truths is not None:
            print("\nBaseline scorecard:")
            print(score(preds, village))
        return
        
    # 2. Load boundary hints raster
    with rasterio.open(village.boundaries_path) as b_src:
        bounds_data = b_src.read(1)
        b_transform = b_src.transform
        b_bounds = b_src.bounds
        b_crs = b_src.crs
        
    print("Computing Euclidean Distance Transform of boundaries.tif...")
    binary_mask = (bounds_data == 255)
    dt = distance_transform_edt(~binary_mask)
    
    # 3. Compute starting global median shift coordinates from training baseline
    base_preds = global_median_shift(village)
    note = base_preds['method_note'].iloc[0]
    
    m = re.search(r"dx=([\d.-]+)m dy=([\d.-]+)m", note)
    if m:
        global_dx = float(m.group(1))
        global_dy = float(m.group(2))
    else:
        global_dx, global_dy = 0.0, 0.0
    print(f"Global starting shift baseline: dx={global_dx:.2f}m, dy={global_dy:.2f}m")
    
    left, bottom, right, top = b_bounds
    px_w = b_transform.a
    px_h = b_transform.e
    
    # Reproject all plots to the boundaries CRS (EPSG:3857)
    plots_3857 = village.plots.to_crs(b_crs)
    
    optimized_geoms = []
    confidences = []
    statuses = []
    notes = []
    
    # Local search bounds (e.g. ±15m relative to global shift)
    dx_search = np.arange(global_dx - 15, global_dx + 15.5, 0.5)
    dy_search = np.arange(global_dy - 15, global_dy + 15.5, 0.5)
    
    print(f"Running localized alignment search for {len(village.plots)} plots...")
    for idx, pn in enumerate(village.plots.index):
        geom_3857 = plots_3857.loc[pn, 'geometry']
        pts = get_boundary_points(geom_3857, step_m=2.0)
        
        if len(pts) == 0:
            # Fallback to global median shift if no points are extractable
            shifted_geom = translate(geom_3857, global_dx, global_dy)
            optimized_geoms.append(shifted_geom)
            confidences.append(0.1)
            statuses.append('flagged')
            notes.append('no boundary points; fallback to global shift')
            continue
            
        X = pts[:, 0]
        Y = pts[:, 1]
        
        N = len(pts)
        NDX = len(dx_search)
        NDY = len(dy_search)
        
        # Calculate row and column indices in boundaries.tif
        cols = (X[:, None, None] + dx_search[None, :, None] - left) / px_w
        rows = (top - (Y[:, None, None] + dy_search[None, None, :])) / abs(px_h)
        
        # Broadcast indices to align coordinate arrays
        rows, cols = np.broadcast_arrays(rows, cols)
        
        coords = np.vstack([rows.ravel(), cols.ravel()])
        # Sample EDT values using bilinear interpolation (order=1)
        d_vals = map_coordinates(dt, coords, order=1, mode='constant', cval=999.0)
        d_vals = d_vals.reshape(N, NDX, NDY)
        
        # Loss is average pixel distance of shape perimeter to edge lines
        loss_grid = np.mean(d_vals, axis=0)
        
        min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
        best_dx = dx_search[min_idx[0]]
        best_dy = dy_search[min_idx[1]]
        
        # 4. Fine Search stage (0.1m resolution)
        dx_fine = np.arange(best_dx - 0.8, best_dx + 0.9, 0.1)
        dy_fine = np.arange(best_dy - 0.8, best_dy + 0.9, 0.1)
        
        cols_fine = (X[:, None, None] + dx_fine[None, :, None] - left) / px_w
        rows_fine = (top - (Y[:, None, None] + dy_fine[None, None, :])) / abs(px_h)
        
        rows_fine, cols_fine = np.broadcast_arrays(rows_fine, cols_fine)
        coords_fine = np.vstack([rows_fine.ravel(), cols_fine.ravel()])
        
        d_vals_fine = map_coordinates(dt, coords_fine, order=1, mode='constant', cval=999.0)
        d_vals_fine = d_vals_fine.reshape(N, len(dx_fine), len(dy_fine))
        
        loss_grid_fine = np.mean(d_vals_fine, axis=0)
        min_idx_fine = np.unravel_index(np.argmin(loss_grid_fine), loss_grid_fine.shape)
        
        best_dx = dx_fine[min_idx_fine[0]]
        best_dy = dy_fine[min_idx_fine[1]]
        min_loss = loss_grid_fine[min_idx_fine]
        
        # Translate the geometry using optimized shift
        shifted_geom = translate(geom_3857, best_dx, best_dy)
        optimized_geoms.append(shifted_geom)
        
        # Calibrate confidence based on how close the boundary is to binary lines
        # An average distance of <= 6 pixels scales to confidence in [0, 1]
        conf = float(np.clip(1.0 - min_loss / 6.0, 0.0, 1.0))
        
        # If we have a relatively clear alignment, mark as corrected, otherwise flag
        if conf > 0.4:
            statuses.append('corrected')
        else:
            statuses.append('flagged')
            
        confidences.append(conf)
        notes.append(f"local optimization dx={best_dx:.2f}m dy={best_dy:.2f}m loss={min_loss:.2f}")
        
        if (idx + 1) % 500 == 0:
            print(f"  aligned {idx + 1}/{len(village.plots)} plots...")
            
    # 5. Save results to predictions.geojson
    preds = village.plots.copy()
    temp_gdf = gpd.GeoDataFrame(geometry=optimized_geoms, index=village.plots.index, crs=b_crs)
    preds['geometry'] = temp_gdf.to_crs('EPSG:4326').geometry
    preds['status'] = statuses
    preds['confidence'] = confidences
    preds['method_note'] = notes
    
    out = write_predictions(Path(village_dir) / 'predictions.geojson', preds)
    print(f"Successfully wrote {len(preds)} predictions to {out}")
    
    if village.example_truths is not None:
        print("\n=== Optimized scorecard vs example truths ===")
        print(score(preds, village))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VILLAGE)
