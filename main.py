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

plt.rcParams["font.family"] = "Times New Roman"
shapely.speedups.enable()
warnings.filterwarnings('ignore')
%matplotlib inline

cluster_spots, cluster_time = [],[]
static_frames, moving_frames = [], []
static_frames_t, moving_frames_t = [], []
lm_list = []
lm_moves_list = []
# txts = []
times = []
moving_frame_t = []
moving_cluster_nps = []

# directory = '/Users/GeorgePyne/Documents/CASA/Dissertation/IPYNB/decompSpaceTime-master/pointFilesworking9Jun/'
directory = 'pointFiles'


static_frames, moving_frames = [], [] # Store cluster geometries
static_frames_t, moving_frames_t = [], [] # Store cluster times
lm_list,lm_moves_list = [], [] # Store lm objects and lm dfs
run_times = [] # Store runtimes


directory = '/Users/GeorgePyne/Documents/CASA/Dissertation/IPYNB/decompSpaceTime-master/pointFiles'
count = 0
length = len(os.listdir(directory))

for filename in os.listdir(directory): # Iterate over each decomposition domain
    txt = pd.read_csv(directory+'/'+filename, skiprows=1, names=['lon','lat','time']) # read decomp
    txt = txt.sort_values(by='time')
    points = [[lat,lon] for lat,lon in zip(txt.lat, txt.lon)] # Save point patter to test global CSR
    txt['geometry'] = [shapely.geometry.Point(p) for p in points]
    txt = GeoDataFrame(txt)
    pp = PointPattern(points) # Create point pattern object
    domain_quadrat = qs.QStatistic(pp, shape= "rectangle", nx = 6, ny = 7) # Run quadrat analysis on point pattern object
    pv = float(str(domain_quadrat.chi2_pvalue)[0:4])
    if pv < 1.0: # If not CSR and if statistically significant
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

        W = ps.weights.Queen.from_dataframe(grid) # Calculate spatial Queen contiguity weights object from grid gdf 
        transitions = np.array(grid[grid.columns[2:]]) # Save transitions as matrix
        lm = ps.LISA_Markov(transitions,W) # Calculate LISA markov transitions from W and matrix
        lm_list.append(lm)
        lm_moves = lm_to_df(lm) # creat a LISA transition df of observed against expected
        lm_moves_list.append(lm_moves)


        frame_t, cluster_nps, cluster_hull = [],[],[]
        
        for frame in frames:                
            if len(frame)>4:
                points = [[lat,lon] for lat,lon in zip(frame.lat, frame.lon)]
                pp_t = PointPattern(points)
                stdd = std_distance(pp_t.points)
                q_h_t = qs.QStatistic(pp,shape= "rectangle",nx = 6, ny = 7)
                pv_t = float(str(q_h_t.chi2_pvalue)[0:4])
                if pv_t < 1.0: 
                    clust = OPTICS(min_cluster_size=4).fit(points)
                    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=stdd)
                    frame['labels'] = labels
                    for i in frame.labels.unique():
                        if i > -1:
                            if frame.labels.value_counts()[i] > 3:
                                geom = frame.loc[frame['labels']==i]
                                absortion_condition = PointPattern(zip(geom.lon.to_list(), geom.lat.to_list())).mean_nnd
                                cluster_hull.append\
                                (shapely.geometry.MultiPoint(geom.geometry.tolist()).convex_hull.buffer(distance=absortion_condition))
                                frame_t.append(frame.time.unique()[0])
                
        if lm_moves['Residuals']['LH-HH'] > 0:
            moving_frames = moving_frames+cluster_hull
            moving_frames_t = moving_frames_t+frame_t
        else:
            static_frames = static_frames+cluster_hull
            static_frames_t = static_frames_t+frame_t
        end_time = datetime.now() # Save endtime
        run_time = end_time - start_time # Find runtime to adjudge parallelization
        run_times.append(run_time)
    else:
         print("CSR expected in {}.".format(filename))
    count = count + 1
    print(f'Count at {count} of {length}.')
