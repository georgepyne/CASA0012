import os
from datetime import datetime, timedelta
from geopandas import GeoDataFrame
import geopandas as gpd
import gpd_lite_toolbox as glt
import pointpats.quadrat_statistics as qs
from pointpats import PointPattern
import pysal as ps
import shapely.speedups
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
import shapely.geometry
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import contextily as ctx
import zipfile
from shapely.strtree import STRtree
from pointpats.centrography import std_distance


static_frames, moving_frames = [], [] # Store cluster geometries
static_frames_t, moving_frames_t = [], [] # Store cluster times
lm_list,lm_moves_list = [], [] # Store lm objects and lm dfs
run_times = [] # Store runtimes
lm_times = []
grid_times = []
cluster_times = []
lm_record = 0

directory = '/Users/GeorgePyne/Documents/CASA/Dissertation/IPYNB/decompSpaceTime-master/pointFiles'
count = 0
length = len(os.listdir(directory))

for filename in os.listdir(directory)[2:17]: # Iterate over each decomposition domain
    txt = pd.read_csv(directory+'/'+filename, skiprows=1, names=['lon','lat','time']) # read decomp
    txt = txt.sort_values(by='time')
    points = [[lat,lon] for lat,lon in zip(txt.lat, txt.lon)] # Save point patter to test global CSR
    txt['geometry'] = [shapely.geometry.Point(p) for p in points]
    txt = GeoDataFrame(txt)
    pp = PointPattern(points) # Create point pattern object
    domain_quadrat = qs.QStatistic(pp, shape= "rectangle", nx = 6, ny = 7) # Run quadrat analysis on point pattern object
    pv = float(str(domain_quadrat.chi2_pvalue)[0:4])
    if pv < 0.5: # If not CSR and if statistically significant
        start_time = datetime.now() # Time subdomain process

        # Create geo_df of points to rasterize
        grid_xmin,grid_ymin,grid_xmax,grid_ymax = txt.total_bounds # Get point bounds
        height = (grid_xmax-grid_xmin)/ 6 # Set grid resolution
        grid = glt.make_grid(txt,height, False).reset_index().rename(columns={'index':'Cell ID'})# Create unclipped grid
        dfsjoin = gpd.sjoin(grid, txt) #Spatial join Points to polygons
        dfpivot = pd.pivot_table(dfsjoin,index='Cell ID',columns='time',aggfunc={'time':len})
        dfpivot.columns = dfpivot.columns.droplevel()
        grid = (grid.merge(dfpivot, how='left',on='Cell ID')).fillna(0)

        frames = [txt.loc[txt.time == t] for t in txt.time.unique()]
        update = datetime.now()
        print(f'Created grid in {update - start_time}.')
        grid_times.append(update - start_time)

        W = ps.weights.Queen.from_dataframe(grid) # Calculate spatial Queen contiguity weights object from grid gdf
        transitions = np.array(grid[grid.columns[2:]]) # Save transitions as matrix
        lm = ps.LISA_Markov(transitions,W) # Calculate LISA markov transitions from W and matrix
        lm_list.append(lm)
        lm_moves = lm_to_df(lm) # creat a LISA transition df of observed against expected
        lm_moves_list.append(lm_moves)

        lm_time = datetime.now()
        print(f'LM calculated in {lm_time - update}.')
        lm_times.append(lm_time - update)

        frame_t, cluster_nps, cluster_hull = [],[],[]

        for frame in frames:
            if len(frame)>4: # For polygonal geometry
                points = [[lat,lon] for lat,lon in zip(frame.lat, frame.lon)]
                pp_t = PointPattern(points) # pp for local csr
                stdd = std_distance(pp_t.points) # Calculate stdd for absorption conition
                q_h_t = qs.QStatistic(pp,shape= "rectangle",nx = 6, ny = 7) # Check local CSR
                pv_t = float(str(q_h_t.chi2_pvalue)[0:4])
                if pv_t < 0.5: # Statistically significant CSR
                    clust = OPTICS(min_cluster_size=4).fit(points) # Run OPTICS
                    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=stdd) # Extract cluster labels
                    frame['labels'] = labels
                    for i in frame.labels.unique():
                        if i > -1:
                            if frame.labels.value_counts()[i] > 3:
                                geom = frame.loc[frame['labels']==i]
                                cluster_hull.append\
                                (shapely.geometry.MultiPoint(geom.geometry.tolist()).convex_hull) # save clusterhull for each label
                                frame_t.append(frame.time.unique()[0]) # Save clustertime for t index

        if lm_moves['Residuals']['LH-HH'] > lm_record: # Check for MCCC condition
            lm_record = lm_moves['Residuals']['LH-HH']
            LISA_markov_diagnostics(lm_moves)

            moving_frames = moving_frames+cluster_hull # Save clusterhull geometry
            moving_frames_t = moving_frames_t+frame_t # Save clusterhull time
        else:
            static_frames = static_frames+cluster_hull
            static_frames_t = static_frames_t+frame_t

        cluster_time = datetime.now()
        print(f'Cluster extraction calculated in {cluster_time - lm_time}.')
        cluster_times.append(cluster_time - lm_time)



        end_time = datetime.now() # Save endtime
        run_time = end_time - start_time # Find runtime to adjudge parallelization
        run_times.append(run_time)
    else:
        print(f"CSR expected in {filename}.")

def lm_to_df(lm):
        lm_moves = pd.DataFrame({"Transitions":lm.transitions.flatten(), "Expected":lm.expected_t.flatten()})
        lm_moves['Transitions'] = lm_moves['Transitions'].astype(int)
        lm_moves.index = ["HH-HH","HH-LH","HH-LL","HH-HL", # Make index 16 possible move types
                  "LH-HH","LH-LH","LH-LL","LH-HL",
                  "LL-HH","LL-LH","LL-LL","LL-HL",
                  "HL-HH","HL-LH","HL-LL","HL-HL"]
        lm_moves['Residuals'] = lm_moves.Transitions - lm_moves.Expected # Calculate difference of observed - expected


        return lm_moves


def create_index(cluster_times, cluster_geometries):

    """Creates a key-value pair global-local time-space index
    with time as the key lookup and values are individual STR-trees"""

    lookup = {} # create dict for key-value lookup
    for ct, cg in zip(cluster_times, cluster_geometries):
        if ct in lookup: # Check if STR-tree is drawn for t
            lookup[ct] = STRtree(lookup[ct]._geoms+[cg]) # Redraw STR_tree if record exists
        else:
            lookup[ct] = STRtree([cg]) # Create STR-tree from geometry list

    return lookup

def mc_poly_gdf(index):
    times,geometries = [],[]
    for t, g in index.items():
            geoms = g._geoms
            geoms = [shape for shape in geoms if type(shape) == shapely.geometry.polygon.Polygon]
            geoms = [shapely.geometry.Polygon(zip(geom.exterior.coords.xy[1],geom.exterior.coords.xy[0])) for geom in geoms]
            if len(geoms) > 1:
                geometries.append(shapely.geometry.MultiPolygon(geoms))
                times.append(t)
            elif geoms:
                geometries.append(geoms[0])
                times.append(t)

    mc_poly_gdf = GeoDataFrame({'time':times, 'geometry':geometries})

    return mc_poly_gdf

def extract_moving_cluster_object(data, str_tree):
    results = []
    data['geometry_query'] = [shapely.geometry.Point(lat,lon) for lat,lon in zip(data.Latitude,data.Longitude)]
    for geom,time in zip(data['geometry_query'],data['time']):
        try:
            result = str_tree[time].query(geom)
            if result:
                results.append(True)
            else:
                results.append(False)
        except:
            results.append(False)
    data['Intersect'] = results
    data = data.drop('geometry_query',axis=1)
    data = data.loc[data['Intersect']==True].drop('Intersect', axis=1)

    return data
