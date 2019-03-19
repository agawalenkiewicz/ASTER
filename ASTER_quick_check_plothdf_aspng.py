import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyhdf.SD import SD, SDC
from math import radians, cos, sin, asin, sqrt, atan, tan
import re
from pyhdf.error import HDF4Error
from datetime import datetime


filename = sys.argv[1]
#Read file and look at contents
hdf = SD(filename, SDC.READ)
print hdf.datasets()

def get_calendardate(hdf):
	hdf_attributes = hdf.attributes()
	coremetadata =  hdf_attributes.get('coremetadata.0')
	start_index = coremetadata.find("OBJECT                 = CALENDARDATE")
	end_index = coremetadata.find(" END_OBJECT             = CALENDARDATE")
	my_string = coremetadata[start_index:end_index]
	my_calendardate = re.findall(r'\b(\d{8})\b' , my_string)
	dated = datetime.strptime(my_calendardate[0], '%Y%m%d')
	day = dated.timetuple()
	doy = day.tm_yday
	return doy

def get_solar_elev(hdf):
	productmetadata = hdf_attributes.get('productmetadata.0')
	start_index = productmetadata.find("OBJECT                 = SOLARDIRECTION")
	end_index = productmetadata.find("END_OBJECT             = SOLARDIRECTION")
	my_string = productmetadata[start_index:end_index]
	nums = re.compile(r"[+-]?\d+(?:\.\d+)")
	azimuth, elevation = re.findall(nums, my_string)
	elev = np.float(elevation)
	return elev

def get_gainInfo(hdf):
	productmetadata = hdf_attributes.get('productmetadata.0')
	start_index = productmetadata.find("GROUP                  = GAININFORMATION")
	end_index = productmetadata.find("END_GROUP              = GAININFORMATION")
	my_string = productmetadata[start_index:end_index]
	gains = re.findall(r'(?<=\().*?(?=\))', my_string)
	band_list = []
	gain_list = []
	for item in gains:
		band, gain = map(str, item.split(','))
		band_list.append(band)
		gain_list.append(gain)
	output_dict = dict(zip(keys, values))
	return output_dict


datafield_name = 'ImageData11'
# Read dataset.
data2D = hdf.select(datafield_name)
data = data2D[:,:].astype(np.double)
print np.shape(data)

# Create geolocation dataset from HDF-EOS2 dumper output.
LAT_GEO_FILE_NAME = sys.argv[2]
lat = np.genfromtxt(LAT_GEO_FILE_NAME, delimiter=',', usecols=[0])
print np.shape(lat)
lat = lat.reshape(data.shape)

LON_GEO_FILE_NAME = sys.argv[3]
lon = np.genfromtxt(LON_GEO_FILE_NAME, delimiter=',', usecols=[0])
lon = lon.reshape(data.shape)

# Limit map based on min/max lat/lon values because the file covers a small region.
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=np.min(lat), urcrnrlat=np.max(lat),
            llcrnrlon=np.min(lon), urcrnrlon=np.max(lon))            
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(np.floor(np.min(lat)), np.ceil(np.max(lat)), 1), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(np.floor(np.min(lon)), np.ceil(np.max(lon)), 1), labels=[0, 0, 0, 1])

m.pcolormesh(lon, lat, data)
cb = m.colorbar()

basename = os.path.basename(filename)
plt.title('{0}\n{1}\n'.format(basename, datafield_name), fontsize=11)
    
fig = plt.gcf()
pngfile = "{0}.py.png".format(basename)
fig.savefig(pngfile)


"""
../../../../../../data/hdfeos/eos2dump -a2 
hartlepool/2169414789/AST_L1T_00306112015112217_20150615161750_8917.hdf 
TIR_Swath > hartlepool/2169414789/TIR_lon.out
"""