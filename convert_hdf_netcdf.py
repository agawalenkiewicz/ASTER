import netCDF4 as nc
import time
import numpy as np

def create_netcdf(output_name, regridded_data_3d, regridded_lat_array, regridded_lon_array):
	#open a new netcdf file in the write ('w') mode
	dataset = nc.Dataset(output_name+'.nc', 'w', format='NETCDF4_CLASSIC')

	#create dimensions
	#band = dataset.createDimension('band', bands_len) #11 can later be not hard coded but a variable e.g. band_number
	lat = dataset.createDimension('lat', len(regridded_lat_array)) #400 not hard coded but e.g. len(lat_array) of the regridded dataset
	lon = dataset.createDimension('lon', len(regridded_lon_array)) #400 not hard coded but e.g. len(lon_array) of the regridded dataset
	time = dataset.createDimension('time', None) #unlimited time, if we want to add data later

	# Create coordinate variables for 4-dimensions
	time = dataset.createVariable('time', np.float32, ('time',))
	time.standard_name = 'time'
	#bands = dataset.createVariable('bands', np.int32, ('band',))
	latitudes = dataset.createVariable('latitude', np.float32, ('lat','lon',))
	latitudes.standard_name = 'latitude'
	longitudes = dataset.createVariable('longitude', np.float32, ('lat','lon',))
	longitudes.standard_name = 'longitude'

	# Create the actual 4-d variable
	BT10 = dataset.createVariable('BT_band10', np.float32, ('lat','lon'))
	BT11 = dataset.createVariable('BT_band11', np.float32, ('lat','lon'))
	BT12 = dataset.createVariable('BT_band12', np.float32, ('lat','lon'))
	BT13 = dataset.createVariable('BT_band13', np.float32, ('lat','lon'))
	BT14 = dataset.createVariable('BT_band14', np.float32, ('lat','lon'))
	reflectance1 = dataset.createVariable('reflectance_band1', np.float32, ('lat','lon'))
	reflectance2 = dataset.createVariable('reflectance_band2', np.float32, ('lat','lon'))
	reflectance3N = dataset.createVariable('reflectance_band3N', np.float32, ('lat','lon'))


	#Create Global Attributes
	#dataset.description = 'Landsat 8 regridded data'
	#dataset.history = 'Created ' + time.ctime(time.time())

	# Variable Attributes
	latitudes.units = 'degree_north'
	longitudes.units = 'degree_east'
	#bands.units = 'micro_meters'
	BT10.units = 'K'
	BT11.units = 'K'
	BT12.units = 'K'
	BT13.units = 'K'
	BT14.units = 'K'
	#reflectance1.units = 'Watts_meter^-2'
	#reflectance2.units = 'Watts_meter^-2'
	#reflectance3N.units = 'Watts_meter^-2'
	
	#Assign values to the variables
	lats = regridded_lat_array #np.arange(-90,91,2.5)
	lons = regridded_lon_array #np.arange(-180,180,2.5)
	latitudes[:,:] = lats[:,:]
	longitudes[:,:] = lons[:,:]
	print(latitudes)
	print(longitudes)
	reflectance1[:,:] = regridded_data_3d[0,:,:]
	reflectance2[:,:] = regridded_data_3d[1,:,:]
	reflectance3N[:,:] = regridded_data_3d[2,:,:]

	BT10[:,:] = regridded_data_3d[3,:,:] #[0,:,:]
	BT11[:,:] = regridded_data_3d[4,:,:] #[1,:,:]
	BT12[:,:] = regridded_data_3d[5,:,:] #[2,:,:]
	BT13[:,:] = regridded_data_3d[6,:,:] #[3,:,:]
	BT14[:,:] = regridded_data_3d[7,:,:] #[4,:,:]
	#print(BT10)
	#print(BT11.shape)
	
	dataset.close()
	return
