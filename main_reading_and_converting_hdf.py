import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyhdf.SD import SD, SDC
import math
from math import radians, cos, sin, asin, sqrt, atan, tan
import re
from pyhdf.error import HDF4Error
from datetime import datetime

from scipy.interpolate import griddata
from scipy.interpolate import interpolate

import proj_grid_for_hdf as p
import regridding_hdf as rh
import convert_hdf_netcdf as conv_nc


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
		band, gain = map(str, item.split(', '))
		band_list.append(band)
		gain_list.append(gain)
	output_dict = dict(zip(band_list, gain_list))
	return output_dict

def get_lines_samples(str):
	#get starting index of the substring - number of spaces is important! DO NOT CHANGE!
	start_index = str.find("OBJECT                 = IMAGEDATAINFORMATION")
	#get end index of the substring
	end_index = str.find("END_OBJECT             = IMAGEDATAINFORMATION")
	# because the start and end indices are variables based on text,
	# you can extract the right substring each time
	# by passing the start and end index as variables
	my_string = str[start_index:end_index]
	# my_string contains information only about the corner cooridnates
	# use regular expressions to find numbers 
	#and change stirng into a float ot integer
	for match in re.findall(r'(?<=\().*?(?=\))',my_string):
		pixels, lines, BPP = map(float,match.split(','))
		#print pixels, lines, BPP
	return pixels, lines, BPP

def get_corner_coord(str):
	#get starting index of the substring - number of spaces is important! DO NOT CHANGE!
	start_index = str.find("GROUP                  = SCENEFOURCORNERS")
	#get end index of the substring
	end_index = str.find("END_GROUP              = SCENEFOURCORNERS")
	# because the start and end indices are variables based on text,
	# you can extract the right substring each time
	# by passing the start and end index as variables
	my_string = str[start_index:end_index]
	# my_string contains information only about the corner cooridnates
	# use regular expressions to find numbers 
	#BUT also save the sign in front of numbers (in case of negative numbers)
	nums = re.compile(r"[+-]?\d+(?:\.\d+)")
	re.findall(nums, my_string)
	# save the list of STRINGS as your list you corner lats and lons
	corner_lats_lons = re.findall(nums, my_string)
	#now each lat and lon can be accessed by it's index
	# UL - upper left , UR - upper right , LL - lower left , LR - lower right
	# because they were strings, they need to be changed to floats
	UL_lat = float(corner_lats_lons[0])
	UL_lon = float(corner_lats_lons[1])
	UR_lat = float(corner_lats_lons[2])
	UR_lon = float(corner_lats_lons[3])
	LL_lat = float(corner_lats_lons[4])
	LL_lon = float(corner_lats_lons[5])
	LR_lat = float(corner_lats_lons[6])
	LR_lon = float(corner_lats_lons[7])
	return (UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon)

	
def haversine(lat1, lat2, lon1, lon2, meters=False):
	"""
	Upper left and upper right lat and lon values need to be in decimal degrees.
	While calculating distance takes into account Earth's curvature.
	Returns distance between two points in km, or m.
	Input: four values as floats, containing the latitude and longitude of each point
	in decimal degrees.
	Output: Returns the distance bewteen the two points.
	The default unit is kilometers. Miles can be returned
	if the ''meters'' parameter is set to True.
	"""
	AVG_EARTH_RADIUS = 6371  # in km
	# convert all latitudes/longitudes from decimal degrees to radians
	lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
	# calculate haversine
	lat = lat2 - lat1
	lon = lon2 - lon1
	h = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lon * 0.5) ** 2
	distance = 2 * AVG_EARTH_RADIUS * asin(sqrt(h))
	
	dist_rad = 2 * asin(sqrt(h))
	#in decimal degrees
	dist_deg = dist_rad * (180 / (np.pi))
	
	#h = hav(UL_lat - UR_lat) + ((np.cos(UR_lat))*(np.cos(UL_lat))*(hav(UL_lon-UR_lon)))
	#distance = 2 * Earth_radius * np.arcsin(np.sqrt(h))
	if meters:
		return distance * 1000  # in meters
	else:
		return distance , dist_deg # in kilometers

		
def latlon_all_pix(band_pixel_line, distance_deg, corner_latlon):
	"""
	This function calculates the lat and lon value for each pixel
	located on the first line and first left and right column
	of the image.
	Input: number of lines or pixels of the image (dependent on band),
	   distance between two corner points calculated from haversine
	   formula, corner value of the first point.
	Return: array of latitude or longitude values for all pixels on the line
	"""
	column_line_pix = np.zeros(band_pixel_line + 1)
	increment = distance_deg / band_pixel_line
	for i in np.arange(0,band_pixel_line + 1,1):
		column_line_pix[i] = corner_latlon - (i * increment)
	return column_line_pix		


def vnir_data(vnir_lat, vnir_lon, attr):
	vnir_bands_names=["Latitude", "Longitude"]
	vnir_bands = []
	if vnir_lat.size <= 3 or vnir_lon.size <= 3:
		pass
	else:
		# Select the right attribute key (a dictionary)
		prod_meta_VNIR = attr['productmetadata.v']
		#get number of pixels and lines for visible and near IR bands
		VNIR_pixel, VNIR_line,VNIR_BPP = get_lines_samples(prod_meta_VNIR)
		try:
			band1 = (hdf.select("ImageData1"))[:,:].astype(np.double)
			vnir_lat = vnir_lat.reshape((band1.shape))
			vnir_lon = vnir_lon.reshape((band1.shape))
			vnir_bands_names = np.append(vnir_bands_names , "ImageData1")
			vnir_bands = np.concatenate([[vnir_lat], [vnir_lon], [band1]])
		except HDF4Error:
			pass
		try:
			band2 = (hdf.select("ImageData2"))[:,:].astype(np.double)
			vnir_bands_names = np.append(vnir_bands_names , "ImageData2")
			vnir_bands = np.concatenate([[vnir_lat], [vnir_lon], [band1], [band2]])
		except HDF4Error:
			pass
		try:
			band3N = (hdf.select("ImageData3N"))[:,:].astype(np.double)
			vnir_bands_names = np.append(vnir_bands_names , "ImageData3N")
			vnir_bands = np.concatenate([[vnir_lat], [vnir_lon], [band1], [band2], [band3N]])
		except HDF4Error:
			pass
		try:
			band3B = (hdf.select("ImageData3B"))[:,:].astype(np.double)
			vnir_bands_names = np.append(vnir_bands_names , "ImageData3B")
			vnir_bands = np.concatenate([[vnir_lat], [vnir_lon], [band1], [band2], [band3N], [band3B]])
		except HDF4Error:
			pass
	return vnir_bands_names, vnir_bands


def swir_data(swir_lat, swir_lon, attr):
	swir_bands_names=["Latitude" , "Longitude"]
	swir_bands=[]
	if swir_lat.size <= 3 or swir_lon.size <= 3 :
		pass
	else:
		# Select the right attribute key (a dictionary)
		prod_meta_SWIR = attr['productmetadata.s']
		#get number of pixels and lines for short wave IR bands
		SWIR_pixel, SWIR_line,SWIR_BPP = get_lines_samples(prod_meta_SWIR)
		try:
			band4 = (hdf.select("ImageData4"))[:,:].astype(np.double)
			swir_lat = swir_lat.reshape((band4.shape))
			swir_lon = swir_lon.reshape((band4.shape))
			swir_bands_names = np.append(swir_bands_names , "ImageData4")
			swir_bands = np.concatenate([[swir_lat], [swir_lon], [band4]])
		except HDF4Error:
			pass
		try:
			band5 = (hdf.select("ImageData5"))[:,:].astype(np.double)
			swir_bands_names = np.append(swir_bands_names , "ImageData5")
			swir_bands = np.concatenate([[swir_lat], [swir_lon], [band4], [band5]])
		except HDF4Error:
			pass
		try:
			band6 = (hdf.select("ImageData6"))[:,:].astype(np.double)
			swir_bands_names = np.append(swir_bands_names , "ImageData6")
			swir_bands = np.concatenate([[siwr_lat], [swir_lon], [band4], [band5], [band6]])
		except HDF4Error:
			pass
		try:
			band7 = (hdf.select("ImageData7"))[:,:].astype(np.double)
			swir_bands_names = np.append(swir_bands_names , "ImageData7")
			swir_bands = np.concatenate([[swir_lat], [swir_lon], [band4], [band5], [band6], [band7]])
		except HDF4Error:
			pass
		try:
			band8 = (hdf.select("ImageData8"))[:,:].astype(np.double)
			swir_bands_names = np.append(swir_bands_names , "ImageData8")
			swir_bands = np.concatenate([[swir_lat], [swir_lon], [band4], [band5], [band6], [band7], [band8]])
		except HDF4Error:
			pass
		try:
			band9 = (hdf.select("ImageData9"))[:,:].astype(np.double)
			swir_bands_names = np.append(swir_bands_names , "ImageData9")
			swir_bands = np.concatenate([[swir_lat], [swir_lon], [band4], [band5], [band6], [band7], [band8], [band9]])
		except HDF4Error:
			pass
	return swir_bands_names, swir_bands

def tir_data(tir_lat, tir_lon, attr):
	tir_bands_names=["Latitude" , "Longitude"]
	tir_bands=[]
	if tir_lat.size <= 3 or tir_lon.size <= 3 :
		pass
	else:
		# Select the right attribute key (a dictionary)
		prod_meta_TIR = attr['productmetadata.t']
		#get number of pixels and lines for thermal bands
		TIR_pixel, TIR_line, TIR_BPP = get_lines_samples(prod_meta_TIR)
		try:
			band10 = (hdf.select("ImageData10"))[:,:].astype(np.double)
			#tir_lat = tir_lat.reshape((band10.shape))
			#tir_lon = tir_lon.reshape((band10.shape))
			tir_bands_names = np.append(tir_bands_names , "ImageData10")
			tir_bands = np.concatenate([[tir_lat], [tir_lon], [band10]])
		except HDF4Error:
			pass
		try:
			band11 = (hdf.select("ImageData11"))[:,:].astype(np.double)
			tir_bands_names = np.append(tir_bands_names , "ImageData11")
			tir_bands = np.concatenate([[tir_lat], [tir_lon], [band10], [band11]])
		except HDF4Error:
			pass
		try:
			band12 = (hdf.select("ImageData12"))[:,:].astype(np.double)
			tir_bands_names = np.append(tir_bands_names , "ImageData12")
			tir_bands = np.concatenate([[tir_lat], [tir_lon], [band10], [band11], [band12]])
		except HDF4Error:
			pass
		try:
			band13 = (hdf.select("ImageData13"))[:,:].astype(np.double)
			tir_bands_names = np.append(tir_bands_names , "ImageData13")
			tir_bands = np.concatenate([[tir_lat], [tir_lon], [band10], [band11], [band12], [band13]])
		except HDF4Error:
			pass
		try:
			band14 = (hdf.select("ImageData14"))[:,:].astype(np.double)
			tir_bands_names = np.append(tir_bands_names , "ImageData14")
			tir_bands = np.concatenate([[tir_lat], [tir_lon], [band10], [band11], [band12], [band13], [band14]])
		except HDF4Error:
			pass
	return tir_bands_names, tir_bands


    
def get_ucc(gain_dict,band):
# Start conversions by band (1-9)  
    if band == 'ImageData1':
        bn = -1 + 1	
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"01"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"01"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        else:
            ucc1 = ucc[bn, 2] 
                        
    if band == 'ImageData2':
        bn = -1 + 2
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"02"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"02"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        else:
            ucc1 = ucc[bn, 2] 
                        
    if band == 'ImageData3N':
        bn = -1 + 3		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"3N"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"3N"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        else:
            ucc1 = ucc[bn, 2] 
                        
    if band == 'ImageData4':
        bn = -1 + 4		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"04"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"04"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        elif gain_dict['"04"'] == '"LO1"':
            ucc1 = ucc[bn, 2] 
        else:
            ucc1 = ucc[bn, 3] 
                        
    if band == 'ImageData5':
        bn = -1 + 5		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"05"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"05"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        elif gain_dict['"05"'] == '"LO1"':
            ucc1 = ucc[bn, 2] 
        else:
            ucc1 = ucc[bn, 3] 
                        
    if band == 'ImageData6':
        bn = -1 + 6  		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"06"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"06"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        elif gain_dict['"06"'] == '"LO1"':
            ucc1 = ucc[bn, 2] 
        else:
            ucc1 = ucc[bn, 3] 
                        
    if band == 'ImageData7':
        bn = -1 + 7  		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"07"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"07"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        elif gain_dict['"07"'] == '"LO1"':
            ucc1 = ucc[bn, 2] 
        else:
            ucc1 = ucc[bn, 3] 
                        
    if band == 'ImageData8':
        bn = -1 + 8 		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"08"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"08"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        elif gain_dict['"08"'] == '"LO1"':
            ucc1 = ucc[bn, 2] 
        else:
            ucc1 = ucc[bn, 3] 
                        
    if band == 'ImageData9':
        bn = -1 + 9  		
        # Query for gain specified in file metadata (by band)            
        if gain_dict['"09"'] == '"HGH"':
            ucc1 = ucc[bn, 0] 
        elif gain_dict['"09"'] == '"NOR"':
            ucc1 = ucc[bn, 1] 
        elif gain_dict['"09"'] == '"LO1"': 
            ucc1 = ucc[bn, 2] 
        else:
            ucc1 = ucc[bn, 3] 
    #else:
        #print "No VNIR or SWIR bands to obtain coefficients"
    return ucc1

def get_irradiance(band):
	if band == 'ImageData1':
		bn = 0
	if band == 'ImageData2':
		bn = 1
	if band == 'ImageData3N':
		bn = 2
	if band == 'ImageData4':
		bn = 3
	if band == 'ImageData5':
		bn = 4
	if band == 'ImageData6':
		bn = 5
	if band == 'ImageData7':
		bn = 6
	if band == 'ImageData8':
		bn = 7
	if band == 'ImageData9':
		bn = 8
	#Set irradiance value for specific band
	irradiance1 = irradiance[bn]
	return irradiance1

def get_tir_coef(band):
	if band == 'ImageData10':
		bn = 0
	if band == 'ImageData11':
		bn = 1
	if band == 'ImageData12':
		bn = 2
	if band == 'ImageData13':
		bn = 3
	if band == 'ImageData14':
		bn = 4
	#Set ucc value for specific band
	tir_ucc1 = tir_ucc_K1_K2[bn, 0]
	K1 = tir_ucc_K1_K2[bn, 1]
	K2 = tir_ucc_K1_K2[bn, 2]
	return tir_ucc1, K1, K2
	
def dn2rad (x, ucc1):
    #ucc1 = np.float(ucc1)
    rad = (x-1.)*ucc1
    return rad
    
def rad2ref (rad, irradiance1):
    ref = (np.pi * rad * (esd * esd)) / (irradiance1 * np.sin(np.pi * sza / 180))
    return ref

def rad2bt(tir_rad1, K1, K2):
	satBT = np.ma.masked_where(tir_rad1 == 0.0, tir_rad1)
	satBT = K1 / tir_rad1
	satBT += 1.0
	satBT = np.log(satBT)
	satBT = K2 / satBT
	return satBT


	

#######################################################################################################
#######################################################################################################

filename = sys.argv[1]
folder = sys.argv[2]

#Read in the file and look at contents
hdf = SD(filename, SDC.READ)
print hdf.datasets()
hdf_attributes = hdf.attributes()
print(hdf_attributes.keys())

#Calculate the day of year (doy) from the calndar date given in the metadata
doy = get_calendardate(hdf)      
# Calculate Earth-Sun Distance    
esd = 1.0 - 0.01672 * np.cos(np.radians(0.9856 * (doy - 4)))
# Need SZA--calculate by grabbing solar elevation info     
sea = get_solar_elev(hdf)
sza = 90 - sea
#Obtain the dictionary of gain values corresponding to each band
gain_dictionary = get_gainInfo(hdf)
#print gain_dictionary
# table for Universal Conversion Coefficients
ucc = np.matrix(([[0.676, 1.688, 2.25, 0.0], [0.708, 1.415, 1.89, 0.0], [0.423, 0.862, 1.15, 0.0], [0.1087, 0.2174, 0.2900, 0.2900], [0.0348, 0.0696, 0.0925, 0.4090], [0.0313, 0.0625, 0.0830, 0.3900], [0.0299, 0.0597, 0.0795, 0.3320], [0.0209, 0.0417, 0.0556, 0.2450], [0.0159, 0.0318, 0.0424, 0.2650]]))
# for TIR bands there is only one possible value for Universal Conversion Coefficients per channel
tir_ucc_K1_K2 = np.matrix(([[0.006822, 3040.136402, 1735.337945], [0.006780, 2482.375199, 1666.398761], [0.006590, 1935.060183, 1585.420044], [0.005693, 866.468575, 1350.069147], [0.005225, 641.326517, 1271.221673]]))
# Thome et al. is used, which uses spectral irradiance values from MODTRAN
# Ordered b1, b2, b3N, b4, b5...b9
irradiance = [1848., 1549., 1114., 225.4, 86.63, 81.85, 74.85, 66.49, 59.85]


#Obtain the latitude and longitude arrays from the files *.out obtained from hdfEOS dumper
vnir_lat_array = np.genfromtxt((folder+"VNIR_lat.out"), usecols=[0], delimiter=",")
vnir_lon_array = np.genfromtxt((folder+"VNIR_lon.out"), usecols=[0], delimiter=",")
#swir_lat_array = np.genfromtxt((folder+"SWIR_lat.out"), usecols=[0], delimiter=",")
#swir_lon_array = np.genfromtxt((folder+"SWIR_lon.out"), usecols=[0], delimiter=",")
tir_lat_array = np.genfromtxt((folder+"TIR_lat.out"), usecols=[0], delimiter=",")
tir_lon_array = np.genfromtxt((folder+"TIR_lon.out"), usecols=[0], delimiter=",")

#Produce a 3D matrix of data for each instrument and an associated list with band names
#vnir_names, vnir_stack = vnir_data(vnir_lat_array, vnir_lon_array, hdf_attributes)
#swir_names, swir_stack = swir_data(swir_lat_array, swir_lon_array, hdf_attributes)
#tir_names, tir_stack = tir_data(tir_lat_array, tir_lon_array, hdf_attributes)



###
# Obtaining a 90m resolution grid for lat and lon of tir data
# Obtaining a 15m resolution grid for lat and lon of vnir data
###
meta0 = hdf_attributes.get('productmetadata.0')
UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon = get_corner_coord(meta0)
#print UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon
metav = hdf_attributes.get('productmetadata.v')
metat = hdf_attributes.get('productmetadata.t')
metas = hdf_attributes.get('productmetadata.s')

pixv, linev, BPPv = get_lines_samples(metav)
pixt, linet, BPPt = get_lines_samples(metat)
pixs, lines, BPPs = get_lines_samples(metas)

horiz_lines_latv = np.zeros((linev, pixv))
lat_horiz_v = np.linspace(UL_lat, UR_lat, pixv)
lat_left_v = np.linspace(UL_lat, LL_lat, linev)
lat_right_v = np.linspace(UR_lat, LR_lat, linev)
for i , (left, right) in enumerate(zip(lat_left_v, lat_right_v)):
	horiz_lines = np.linspace(lat_left_v[i], lat_right_v[i], pixv)
	horiz_lines_latv[i] = horiz_lines
print("horiz_lines_lat_vnir" , horiz_lines_latv)

horiz_lines_lonv = np.zeros((linev, pixv))
lon_horiz_v = np.linspace(UL_lon, UR_lon, pixt)
lon_left_v = np.linspace(UL_lon, LL_lon, linev)
lon_right_v = np.linspace(UR_lon, LR_lon, linev)
for i , (left, right) in enumerate(zip(lon_left_v, lon_right_v)):
	horiz_lines = np.linspace(lon_left_v[i], lon_right_v[i], pixv)
	horiz_lines_lonv[i] = horiz_lines
print("horiz_lines_lon_vnir" , horiz_lines_lonv)

horiz_lines_latt = np.zeros((linet, pixt))
lat_horiz_t = np.linspace(UL_lat, UR_lat, pixt)
lat_left_t = np.linspace(UL_lat, LL_lat, linet)
lat_right_t = np.linspace(UR_lat, LR_lat, linet)
for i , (left, right) in enumerate(zip(lat_left_t, lat_right_t)):
	horiz_lines = np.linspace(lat_left_t[i], lat_right_t[i], pixt)
	horiz_lines_latt[i] = horiz_lines
print("horiz_lines_lat_tir" , horiz_lines_latt)

horiz_lines_lont = np.zeros((linet, pixt))
lon_horiz_t = np.linspace(UL_lon, UR_lon, pixt)
lon_left_t = np.linspace(UL_lon, LL_lon, linet)
lon_right_t = np.linspace(UR_lon, LR_lon, linet)
for i , (left, right) in enumerate(zip(lon_left_t, lon_right_t)):
	horiz_lines = np.linspace(lon_left_t[i], lon_right_t[i], pixt)
	horiz_lines_lont[i] = horiz_lines
print("horiz_lines_lon_tir" , horiz_lines_lont)

horiz_lines_lats = np.zeros((lines, pixs))
lat_horiz_s = np.linspace(UL_lat, UR_lat, pixs)
lat_left_s = np.linspace(UL_lat, LL_lat, lines)
lat_right_s = np.linspace(UR_lat, LR_lat, lines)
for i , (left, right) in enumerate(zip(lat_left_s, lat_right_s)):
	horiz_lines = np.linspace(lat_left_t[i], lat_right_t[i], pixs)
	horiz_lines_lats[i] = horiz_lines
print("horiz_lines_lat_swir" , horiz_lines_lats)

horiz_lines_lons = np.zeros((lines, pixs))
lon_horiz_s = np.linspace(UL_lon, UR_lon, pixs)
lon_left_s = np.linspace(UL_lon, LL_lon, lines)
lon_right_s = np.linspace(UR_lon, LR_lon, lines)
for i , (left, right) in enumerate(zip(lon_left_s, lon_right_s)):
	horiz_lines = np.linspace(lon_left_t[i], lon_right_t[i], pixs)
	horiz_lines_lons[i] = horiz_lines
print("horiz_lines_lon_swir" , horiz_lines_lons)

tir_names, tir_stack = tir_data(horiz_lines_latt, horiz_lines_lont, hdf_attributes)
vnir_names, vnir_stack = vnir_data(horiz_lines_latv, horiz_lines_lonv, hdf_attributes)
swir_names, swir_stack = swir_data(horiz_lines_lats, horiz_lines_lons, hdf_attributes)
#print "tir_stack" , tir_stack
#print "tir_names" , tir_names


# Cerate empty arrays for vnir and tir bands to populate them with radiance and reflectance
vnir_rad = np.zeros((vnir_stack.shape))
vnir_ref = np.zeros((vnir_stack.shape))
swir_rad = np.zeros((swir_stack.shape))
swir_ref = np.zeros((swir_stack.shape))
tir_rad = np.zeros((tir_stack.shape))
tir_BT = np.zeros((tir_stack.shape))
gain_dict = get_gainInfo(hdf)

vnir_rad[0] = vnir_stack[0] #latitude
vnir_rad[1] = vnir_stack[1] #longitude
vnir_ref[0] = vnir_stack[0] #latitude
vnir_ref[1] = vnir_stack[1] #longitude

# Loop through the zipped names and data with indicating which layer it is
# calculate layer by layer (2D arrays) rad and ref
# populate the new 2D arrays with new data
# Output is a 3D array of radiance and reflectance, where each layer is a seperate band
for i, (name, layer) in enumerate(zip(vnir_names[2:], vnir_stack[2:])):
	vnir_ucc = get_ucc(gain_dict, name)
	#print 'vnir ucc is' , vnir_ucc
	vnir_irradiance = get_irradiance(name)
	#print 'vnir irradiance is' , vnir_irradiance
	# Convert from DN to Radiance    
	#vnir_rad[i] = dn2rad(original_data[i])
	vnir_rad[i] = dn2rad(layer, vnir_ucc)
	vnir_rad[i][vnir_rad[i] == dn2rad(0, vnir_ucc)] = 0
	# Convert from Radiance to TOA Reflectance
	vnir_ref[i+2] = rad2ref(vnir_rad[i], vnir_irradiance)

#print 'RADIANCE'
#print vnir_rad[0]
#print vnir_rad[1]
#print vnir_rad[2]
#print 'REFLECTANCE'
#print vnir_ref[0]
#print vnir_ref[1]	
#print vnir_ref[2]

tir_BT[0] = tir_stack[0] #latitude
tir_BT[1] = tir_stack[1] #longitude
# Loop through the zipped names and data with indicating which layer it is
# calculate layer by layer (2D arrays) rad and ref
# populate the new 2D arrays with new data
# Output is a 3D array of radiance and reflectance, where each layer is a seperate band
for i, (name, layer) in enumerate(zip(tir_names[2:], tir_stack[2:])):
	#print name
	#print layer
	tir_ucc, tir_K1, tir_K2 = get_tir_coef(name)
	#print 'tir ucc is' , tir_ucc
	# Convert from DN to Radiance    
	#vnir_rad[i] = dn2rad(original_data[i])
	tir_rad[i] = dn2rad(layer, tir_ucc)
	tir_rad[i][tir_rad[i] == dn2rad(0, tir_ucc)] = 0
	# Convert from Radiance to BT
	tir_BT[i+2] = rad2bt(tir_rad[i], tir_K1, tir_K2)


###



			#current    #Hartlepool #Dungeness #Hinckley #Hunterston-needs fix #Torness   #Sizewell  #Heysham
center_lat = 50.913889 	#54.635     #50.913889 #51.209   #55.726366-needs fix  #55.969752 #52.213461 #54.028889
center_lon = 0.963889  #-1.180833  #0.963889  #-3.127   #-4.898619-needs fix  #-2.397156 #1.625419  #-2.916111
dist_deg = 0.1
dist_deg_smaller = 0.01
spacing = 0.0005
common_grid, min_lon, min_lat, max_lon, max_lat, spacing = p.creategrid(center_lat, center_lon , dist_deg, spacing)
#common_grid_smaller, min_lon, min_lat, spacing = p.creategrid(center_lat, center_lon , dist_deg_smaller, spacing)
print("common grid" , common_grid)
print("common_grid shape", common_grid.shape)



#pdb.set_trace()
if len(vnir_ref) > 0:
	vnir_regridded, count, distance = rh.regridding("VNIR", vnir_ref, min_lon, min_lat, spacing, common_grid.shape)
#if len(swir_stack) > 0:
	#swir_regridded, count, distance = rh.regridding("SWIR", swir_ref, min_lon, min_lat, spacing, common_grid.shape)
if len(tir_BT) > 0:
	#print "TIR BT multi array" , tir_BT
	tir_regridded, count, distance = rh.regridding("TIR", tir_BT, min_lon, min_lat, spacing, common_grid.shape)

for i in range(0, len(tir_regridded)):
	plt.subplot(211)
	plt.imshow(tir_regridded[i])
	plt.title("tir regridded band 10")
	plt.colorbar()
	plt.clim(275,290)
	plt.subplot(212)
	plt.imshow(tir_BT[i+2])
	plt.title("tir before regridding band 10")
	plt.colorbar()
	plt.clim(275,290)
	plt.show()
"""
for i in range(0, len(vnir_regridded)):
	plt.subplot(211)
	plt.imshow(vnir_regridded[i])
	plt.title("vnir regridded")
	plt.colorbar()
	plt.clim()
	plt.subplot(212)
	plt.imshow(vnir_ref[i+2])
	plt.title("vnir before regridding")
	plt.colorbar()
	plt.clim()
	plt.show()
"""
#print "vnir regridded" , vnir_regridded, "vnir max" , np.amax(vnir_regridded)
#print "swir regridded" , swir_regridded
#print "tir regridded" , tir_regridded, "tir max", np.amax(tir_regridded)

plt.subplot(211)
plt.contourf(common_grid[1], common_grid[0], tir_regridded[0], levels=[275,280,285,290])
plt.title("tir regridded band 10")
#plt.colorbar()
#plt.clim(275,290)
plt.subplot(212)
plt.contourf(tir_BT[1], tir_BT[0], tir_BT[3], levels=[275,280,285,290])
plt.title("tir before regridding band 10")
#plt.colorbar()
#plt.clim(275,290)
plt.show()

ASTER_stack = []

try:
	ASTER_stack = np.concatenate([vnir_regridded, swir_regridded, tir_regridded])
except NameError:
	try: #if len(vnir_stack) > 0 and len(tir_stack) > 0:
		ASTER_stack = np.concatenate([vnir_regridded, tir_regridded])
	except NameError:
		#if len(vnir_stack) > 0:
			#ASTER_stack = vnir_regridded
		if len(tir_stack) > 0: #later change elif to if
			ASTER_stack = tir_regridded

#plt.imshow(ASTER_stack[7])
#plt.show()

#print "ASTER_stacks shape" , ASTER_stack.shape
#create a netcdf file

name = sys.argv[3] # e.g. 'AST_L1T_00307242014113352_20150622072320_92683'
conv_nc.create_netcdf(name, ASTER_stack, common_grid[0,:,:], common_grid[1,:,:])

