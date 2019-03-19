import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os, re
from datetime import datetime
import matplotlib



# for TIR bands there is only one possible value for Universal Conversion Coefficients per channel
tir_ucc_K1_K2 = np.matrix(([[0.006822, 3040.136402, 1735.337945], [0.006780, 2482.375199, 1666.398761], [0.006590, 1935.060183, 1585.420044], [0.005693, 866.468575, 1350.069147], [0.005225, 641.326517, 1271.221673]]))

def get_tir_coef(bandname):
	if bandname == 'BT10':
		bn = 0
	if bandname == 'BT11':
		bn = 1
	if bandname == 'BT12':
		bn = 2
	if bandname == 'BT13':
		bn = 3
	if bandname == 'BT14':
		bn = 4
	#Set ucc value for specific band
	tir_ucc1 = tir_ucc_K1_K2[bn, 0]
	K1 = tir_ucc_K1_K2[bn, 1]
	K2 = tir_ucc_K1_K2[bn, 2]
	return tir_ucc1, K1, K2

	
def bt2rad(tir_bt, K1, K2):
	rad = np.ma.masked_where(tir_bt == 0.0, tir_bt)
	rad = K2 / tir_bt
	rad = np.exp(rad)
	rad -= 1.0
	rad = K1 / rad
	return rad

def stack_aster(filename, path):
	element_path = os.path.join(path, filename)
	nc_file = nc.Dataset(element_path)
	BT10 = np.array(nc_file.variables['BT_band10'])
	BT10[BT10 < 273] = np.nan
	tir_ucc, tir_K1, tir_K2 = get_tir_coef('BT10')
	rad10 = bt2rad(BT10, tir_K1, tir_K2)
	rad10[rad10 < 6.2] = np.nan
	#BT11 = np.array(nc_file.variables['BT_band11'])
	#BT12 = np.array(nc_file.variables['BT_band12'])
	#BT13 = np.array(nc_file.variables['BT_band13'])
	#BT14 = np.array(nc_file.variables['BT_band14'])
	my_cmap = matplotlib.cm.hot_r #coolwarm #BuPu #seismic #magma #coolwarm
	my_cmap.set_bad(color='olive', alpha=1.0) #olive #black #khaki #silver
	plt.imshow(BT10)
	plt.show()
	plt.imshow(rad10, cmap=my_cmap)
	plt.colorbar()
	plt.clim()
	plt.show()
	return rad10 #stack_radiance

###################################
###################################

filename = sys.argv[1] #Landsat_ncfiles.heysham 
path = sys.argv[2] #Landsat_ncfiles.heysham_path 
stack_radiance = stack_aster(filename, path)


"""
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
"""	

for i, layer in enumerate(stack_radiance): 
	#meanBT = np.nanmean(layer)
	#print meanBT
	#layer_amplitude = layer - meanBT
	#layer_amplitude[np.isnan(layer_amplitude)] = np.nan #-999 #this is a new added line
	#print layer_amplitude
	#layer_amplitude[landmask] = np.nan #999
	
	#my_cmap = matplotlib.cm.hot_r #coolwarm #BuPu #seismic #magma #coolwarm
	#my_cmap.set_bad(color='tan', alpha=1.0) #olive #black #khaki #silver
	#my_cmap.set_over(color='khaki') #black #olive
	#my_cmap.set_under(color='khaki') #khaki #grey
	plt.imshow(layer)
	plt.colorbar()
	plt.clim()
	plt.show()