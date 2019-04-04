import numpy as np
import matplotlib.pyplot as plt
import math
#import pyresample.geometry

def creategrid(center_lat, center_lon, dist_deg, spacing, mesh=False):
    """
    Output grid within geobounds and specifised cell size.
    Cell_size_deg should be in decimal degrees.
    Return: common grid with center as a specific loaction
    """
    #calculate the max and min  values of lats from the central point
    max_lat = center_lat + dist_deg
    min_lat = center_lat - dist_deg
    #calculate the max and min  values of lats from the central point
    max_lon = center_lon + dist_deg
    min_lon = center_lon - dist_deg - 0.0000000000003
    #calculate the size of the common grid (a square M by M pixels)  
    cell_size_deg = 2 * dist_deg
    
    N_lat = (max_lat - min_lat) / spacing
    N_lon = (max_lon - min_lon) / spacing
    N_lat = int(N_lat)
    print(N_lat)
    N_lon = int(N_lon) #math.ceil(N_lon) #int(N_lon)
    print(N_lon)
    
    lon = np.zeros([1,N_lon+1])
    lat = np.zeros([N_lat+1,1])
    lon_val = np.arange(min_lon, max_lon, spacing)
    lat_val = np.arange(min_lat, max_lat, spacing)
    lon[0,:] = lon_val
    lat[:,0] = lat_val
    print(lat_val.shape)
    print(lon_val.shape)
    
    lon_out = lon
    lat_out = lat
    #lat_out = np.flipud(lat_out)
    #np.meshgrid
    for i in np.arange(0,N_lat,1):
        lat_out = np.concatenate([lat_out,lat],axis=1)
    for j in np.arange(0,N_lon,1):
        lon_out = np.concatenate([lon_out,lon],axis=0)
            

    lat_out = np.flipud(lat_out)  
    common_grid = np.concatenate([[lat_out], [lon_out]])
    
    return common_grid, min_lon, min_lat, max_lon, max_lat, spacing


