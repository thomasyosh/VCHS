import psycopg2
from psycopg2.extras import execute_values
import requests
import urllib.parse
import time
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
import supabase
import numpy as np

load_dotenv()

SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

SUPABASE_DB_URL=f'postgresql://postgres:{SUPABASE_PASSWORD}@db.sinztwikkrlzuhhdfavs.supabase.co:5432/postgres'

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
def generate_full_csv_with_json_spread():
    print("Fetching ALL data from Supabase...")
    
    # --- 2. Fetch ALL rows ---
    all_rows = []
    limit = 1000
    offset = 0
    
    while True:
        response = supabase.table("tree_cases").select("*").range(offset, offset + limit - 1).execute()
        if not response.data: break
        all_rows.extend(response.data)
        offset += limit

    if not all_rows: return

    df = pd.DataFrame(all_rows)
    df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

    # --- 3. Find unique streets missing coordinates ---
    # missing_coords_mask = df['latitude'].isna() | df['longitude'].isna()
    # streets_to_fetch = df[missing_coords_mask]['street'].dropna().unique()
    streets_to_fetch = df['street'].dropna().unique()
    
    # This dictionary stores API HK Grid (x, y) plus converted WGS84 (lat, lng) per street
    # e.g. {"Nathan Road": [{"x": 835000, "y": 820000, "lat": 22.1, "lng": 114.1}, ...]}
    coordinate_mapping = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    # --- 4. Hit the HK APIs to get MULTIPLE points per street ---
    for street in streets_to_fetch:
        try:
            safe_street = urllib.parse.quote(street)
            map_url = f"https://www.map.gov.hk/gs/api/v1.0.0/locationSearch?q={safe_street}"
            map_resp = requests.get(map_url, headers=headers).json()
            
            api_points = []
            
            if map_resp and len(map_resp) > 0:
                # Limit to 5 points per street so we don't spam the geodetic API for long roads
                max_points_to_fetch = min(len(map_resp), 5) 
                
                for i in range(max_points_to_fetch):
                    easting = map_resp[i].get('x')
                    northing = map_resp[i].get('y')
                    
                    if easting is not None and northing is not None:
                        geo_url = f"https://www.geodetic.gov.hk/transform/v2/?inSys=hkgrid&e={easting}&n={northing}"
                        geo_resp = requests.get(geo_url, headers=headers).json()
                        
                        lat = geo_resp.get('wgsLat')
                        lng = geo_resp.get('wgsLong')
                        
                        if lat and lng:
                            api_points.append({
                                "x": easting,      # HK 1980 Grid easting (from map.gov.hk)
                                "y": northing,     # HK 1980 Grid northing (from map.gov.hk)
                                "lat": lat,
                                "lng": lng,
                            })
                
                if api_points:
                    coordinate_mapping[street] = api_points
                    print(f"Mapped {street} across {len(api_points)} distinct points.")
                    
        except Exception as e:
            print(f"Error processing {street}: {e}")
        
        # Polite rate limiting
        time.sleep(0.3)

    # --- 5. Apply coordinates using Round-Robin JSON Spread & Micro-Jitter ---
    print("\nApplying coordinates evenly across API points...")
    
    # A very tiny jitter (roughly 5 meters) just to separate duplicates that land on the exact same API point
    MICRO_JITTER = 0.00005 
    # Roughly equivalent ~5m jitter on HK Grid metres (so stacked cases stay separable in x/y too)
    MICRO_JITTER_XY = 5.0
    
    for street, api_points in coordinate_mapping.items():
        # Get the indices of all rows belonging to this street
        mask = df['street'] == street
        indices = df[mask].index
        num_cases = len(indices)
        num_api_points = len(api_points)
        
        lats = []
        lngs = []
        xs = []
        ys = []
        
        for i in range(num_cases):
            # Round-Robin: Cycle through the available API points
            base_pt = api_points[i % num_api_points]
            
            # Apply micro-jitter only if there are multiple cases to prevent exact stacking
            if num_cases > 1:
                jitter_lat = np.random.uniform(-MICRO_JITTER, MICRO_JITTER)
                jitter_lng = np.random.uniform(-MICRO_JITTER, MICRO_JITTER)
                jitter_x = np.random.uniform(-MICRO_JITTER_XY, MICRO_JITTER_XY)
                jitter_y = np.random.uniform(-MICRO_JITTER_XY, MICRO_JITTER_XY)
            else:
                jitter_lat = 0
                jitter_lng = 0
                jitter_x = 0
                jitter_y = 0
                
            lats.append(base_pt["lat"] + jitter_lat)
            lngs.append(base_pt["lng"] + jitter_lng)
            xs.append(float(base_pt["x"]) + jitter_x)
            ys.append(float(base_pt["y"]) + jitter_y)
            
        # Update the dataframe in one go for this street
        df.loc[indices, 'latitude'] = lats
        df.loc[indices, 'longitude'] = lngs
        df.loc[indices, 'x'] = xs
        df.loc[indices, 'y'] = ys

    # --- 6. Sort by ID and Export ---
    # Ensure geo columns exist even if some streets failed geocoding
    for col in ("latitude", "longitude", "x", "y"):
        if col not in df.columns:
            df[col] = np.nan

    # Keep geo fields together near the end for easier Supabase import review
    preferred_tail = ["latitude", "longitude", "x", "y"]
    other_cols = [c for c in df.columns if c not in preferred_tail]
    df = df[other_cols + preferred_tail]

    csv_filename = 'full_tree_cases_import.csv'
    if 'id' in df.columns:
        df['id'] = df['id'].astype('Int64')
        df = df.sort_values(by='id')
        
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ All done! The spread-out table has been saved to '{csv_filename}'.")
    print("   Included geo columns: latitude, longitude (WGS84), x, y (HK 1980 Grid from map.gov.hk).")
    print("   Before uploading to Supabase, add numeric columns `x` and `y` on table tree_cases if missing.")

if __name__ == "__main__":
    generate_full_csv_with_json_spread()