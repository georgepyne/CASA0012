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
from sklearn.cluster import DBSCAN, OPTICS
import shapely.geometry
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import contextily as ctx
import zipfile

plt.rcParams["font.family"] = "Times New Roman"
shapely.speedups.enable()
warnings.filterwarnings('ignore')
%matplotlib inline

cluster_spots, cluster_time = [],[]
static_frames, moving_frames = [], []
static_frames_t, moving_frames_t = [], []
lm_list = []
lm_moves_list = []
times = []
moving_frame_t = []

def pip_count(points, polygon):
    try :
        pips = points.within(polygon).value_counts()[True]
    except:
        pips = 0

    return pips

directory = '/Users/GeorgePyne/Documents/CASA/Dissertation/IPYNB/decompSpaceTime-master/pointFiles'

for filename in os.listdir(directory)[0:3]: # Iterate over each decomposition domain
    txt = pd.read_csv(directory+'/'+filename, skiprows=1, names=['lon','lat','time']) # read decomp
#     txts.append(txt)
    points = [[lat,lon] for lat,lon in zip(txt.lat, txt.lon)] # Save point patter to test global CSR
    pp = PointPattern(points) # Create point pattern object
    domain_quadrat = qs.QStatistic(pp, shape= "rectangle", nx = 6, ny = 7) # Run quadrat analysis on point pattern object
#     domain_quadrat.plot()
    pv = float(str(domain_quadrat.chi2_pvalue)[0:4])
    if pv < 1.0: # If not CSR and if statistically significant
        start_time = datetime.now() # Time subdomain process

        frames = []
        counts = []

        # Create geo_df of points to rasterize
        points_gdf = GeoDataFrame([shapely.geometry.Point(point) for point in points]).rename(columns={0:'geometry'})
        xmin,ymin,xmax,ymax = points_gdf.total_bounds # Get point bounds
        height = (xmax-xmin)/ 6 # Set grid resolution
        grid = glt.make_grid(points_gdf,height, False) # Create unclipped grid


        for t in txt.time.unique(): # Iterate over each frame
            frame = txt.loc[txt.time == t] # create frame of each time step
            points = [[lat,lon] for lat,lon in zip(frame.lat, frame.lon)] # make list of points
            frame['geometry'] = [shapely.geometry.Point(point) for point in points] # make point df
            frame = GeoDataFrame(frame) # change to gdf
            frames.append(frame)
            grid[str(int(t))+'_count'] = grid.geometry.apply(lambda x: pip_count(frame,x)) # Aggregate points to grid

        W = ps.weights.Queen.from_dataframe(grid) # Calculate spatial Queen contiguity weights object from grid gdf
        tci = np.array(grid[grid.columns[1:]]) # Save transitions as matrix
        lm = ps.LISA_Markov(tci,W) # Calculate LISA markov transitions from W and matrix
        lm_list.append(lm)

        # creat a LISA transition df of observed against expected
        lm_moves = pd.DataFrame({"Transitions":lm.transitions.flatten(), "Expected":lm.expected_t.flatten()})
        lm_moves['Transitions'] = lm_moves['Transitions'].astype(int)
        lm_moves.index = ["HH-HH","HH-LH","HH-LL","HH-HL", # Make index 16 possible move types
                  "LH-HH","LH-LH","LH-LL","LH-HL",
                  "LL-HH","LL-LH","LL-LL","LL-HL",
                  "HL-HH","HL-LH","HL-LL","HL-HL"]
        lm_moves['Residuals'] = lm_moves.Transitions - lm_moves.Expected # Calculate difference of observed - expected
        lm_moves_list.append(lm_moves)

        if lm.chi_2[1] < 0.05:
            for frame in frames:
                if len(frame)>2:
                    points = [[lat,lon] for lat,lon in zip(frame.lat, frame.lon)]
                    pp_t = PointPattern(points)
                    q_h_t = qs.QStatistic(pp,shape= "rectangle",nx = 6, ny = 7)
                    pv_t = float(str(q_h_t.chi2_pvalue)[0:4])
                    if pv_t < 1.0:
                        eps = pp_t.mean_nnd
                        min_samples = int(len(frame) / 6)
                    if min_samples < 3:
                        min_samples = 3
                    labels = OPTICS(eps=eps, min_samples=min_samples).fit(points).labels_
                    frame['labels'] = labels
                    try:
                        for i in frame.labels.unique():
                            if i > -1:
                                if len(frame.loc[frame['labels']==i]) > 2:
                                    moving_frames.append\
                                    (shapely.geometry.MultiPoint\
                                     ([shapely.geometry.Point(lat,lon)\
                                       for lat,lon in zip(frame.loc[frame['labels']==i].lon,\
                                                          frame.loc[frame['labels']==i].lat)]).convex_hull)
                                    moving_frame_t.append(frame.time.unique()[0])
                                    end_time = datetime.now() # Save endtime
                                    run_time = end_time - start_time # Find runtime to adjudge parallelization
                                    times.append(run_time)
                    except:
                        print(frame.head(4))

        else:
            for frame in frames:
                if len(frame)>4:
                    points = [[lat,lon] for lat,lon in zip(frame.lat, frame.lon)]
                    pp_t = PointPattern(points)
                    q_h_t = qs.QStatistic(pp,shape= "rectangle",nx = 6, ny = 7)
                    pv_t = float(str(q_h_t.chi2_pvalue)[0:4])
                    if pv_t < 1.0:
                        eps = pp_t.mean_nnd
                        min_samples = int(len(frame) / 6)
                    if min_samples < 3:
                        min_samples = 3
                    labels = OPTICS(eps=eps, min_samples=min_samples).fit(points).labels_
                    frame['labels'] = labels
                    try:
                        for i in frame.labels.unique():
                            if i > -1:
                                if len(frame.loc[frame['labels']==i]) > 2:
                                    static_frames.append(shapely.geometry.MultiPoint([shapely.geometry.Point(lat,lon) for lat,lon in zip(frame.loc[frame['labels']==i].lon, frame.loc[frame['labels']==i].lat)]).convex_hull)
                                    static_frames_t.append(frame.time.unique()[0])
                                    end_time = datetime.now() # Save endtime
                                    run_time = end_time - start_time # Find runtime to adjudge parallelization
                                    times.append(run_time)
                    except:
                        print(frame.head(4))

        end_time = datetime.now()
        run_time = end_time - start_time
        times.append(run_time)
    else:
        print("CSR expected in {}.".format(filename))
