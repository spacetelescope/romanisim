# gathering of materials for a roll-visibility calculation

#   General note on coordinates used here:
#     Calculations will be performed in Ecliptic coordinates.
#     +X points to vernal equinox; +Z points to North ecliptic pole;
#     +Y makes a right handed coordinate system.
#     Rotating these axes first around +Z to align X' to the longitude
#      of the line of sight, then rotating around Y' to align X'' with
#      the line of sight gives a S/C local set of coordinates:
#     X'' along LOS, Z'' towards increasing latitude,
#     and Y'' pointing towards increasing longitude.
#     Positive roll rotates around X'' according to right hand rule.
#
#   Observatory coordinates:
#     +Xobs is along boresight, +Zobs is normal to Solar array, +Yobs makes right handed sys.
#     At nominal roll, Sun is in Xobs-Zobs plane, with +Zobs as close to Sun as possible.
#     At zero roll offset: +Xobs=+X''  +Yobs=+Y''   +Zobs=+Z''
#     Define position angle of Observatory as angle of +Yobs in degrees East of North.
#
#  LOS = line of sight
#  FOR = Field of Regard (defined by valid LOS angle with respect to the Sun)


# Inputs to this routine:
#   tgt_ra, tgt_dec: coordinates of line of sight (LOS) in RA, DEC.
#   hardwired below to galactic center for a test case
# Outputs from this routine (defined on exiting below) - can wrap this code as desired
#   All outputs are ndarrays of length 365 (or n_samp if weekly or some other variant)
#   nominal_roll,  in degrees; one value for each day of 2028.
#   good_angles,  0 if not in FOR, 1 if in FOR; one value for each day of 2028.
#   sunang_x - angle of Sun with respect to observatory +X axis (boresight), in degrees
#   sunang_y - angle of Sun with respect to observatory +Y axis, in degrees
#   sunang_z - angle of Sun with respect to observatory +Z axis (SASS normal), in degrees
#
#  Jeff Kruk   2021 09 04
#

import numpy as np
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u


# get user RA, DEC by some means - and convert into a SkyCoord object
ra_list = [ '06h00m00s', '08h00m00s', '08h00m00s','17h45m40s', '09h00m00s', '09h00m00s']
dec_list= ['-01d00m00s', '60d00m00s','-60d00m00s','-29d00m28s','89d00m00s','-89d00m00s']
n_tgt=len(ra_list)

ofname = 'test_tgt_vis2.txt'
ofile = open(ofname,'w')
print("Output from test_tgt_vis2.py",file=ofile)
print("Test cases chosen to match corresponding IDL test calculations.",file=ofile)
print("Xang, Yang, Zang are angles of Sun vector in Observatory coordinate frame",file=ofile)
print("X is the boresight, valid angles are 54-126 degrees.",file=ofile)
print("Z is normal to solar array, as close to zero as the pitch allows (<36 at nominal roll when pitch is OK)",file=ofile)
print("Y is perpendicular to X-Z plane; Yang should always be 90 at nominal roll", file=ofile)

# setting up times and Sun coordinates only needs to be done once at the start
# define 1-day intervals, starting Jan 1, 2028
t_start_str = ['2024-01-01T00:00:00.0']
t_start = Time(t_start_str,format='isot', scale='utc')
# if this date is in the future, this will generate a 'dubious date' warning message
# the reason is that it is unknown how many leap seconds will be needed in the future.
# the results will still be valid
#
# Set time sampling as daily over one year
n_days = 365
t_1year = t_start+np.linspace(0.,n_days-1.,n_days)
n_samp = n_days
# To do weekly sampling:
#n_weeks = 52
#n_days = n_weeks*7
#t_1year = t_start+np.linspace(0.,n_days-1.,n_weeks)
# n_samp = n_weeks

# get coordinate object for the Sun for each day of the year
sun_coord = get_body('Sun',t_1year)

# loop over test lines of sight
for i_t in range(n_tgt):

	tgt_ra = ra_list[i_t]
	tgt_dec = dec_list[i_t]

	print("\n\nLine of sight RA  = {0:s}".format(tgt_ra),file=ofile)
	print("Line of sight DEC = {0:s}".format(tgt_dec),file=ofile)
	
	tgt=SkyCoord(tgt_ra, tgt_dec,frame='icrs')

	
	#Get angular separation of LOS to Sun as function of date, in degrees
	sun_angle = sun_coord.separation(tgt)

	min_sun_angle = (90. - 36.) * u.deg
	max_sun_angle = (90. + 36.) * u.deg

	# make array of flags indicating which days are good
	# not getting np.where to behave as I would expect, so do this the old-fashioned way
	good_angles = np.ones(n_samp,dtype='i4')
	for i_d in range(n_samp):
	  if (sun_angle[i_d] < min_sun_angle):
	  	good_angles[i_d] = 0
	  if (sun_angle[i_d] > max_sun_angle):
	  	good_angles[i_d] = 0
	  	
	  	
	#good_angles = np.where( (sun_angle >= min_sun_angle) and (sun_angle <= max_sun_angle), 1, 0)
	# make array of indices - not sure yet if I will use this
	indices = np.arange(n_samp)

	# begin roll angle calculations
	cos_ra_t  = np.cos(tgt.ra.radian)
	cos_dec_t = np.cos(tgt.dec.radian)
	sin_ra_t  = np.sin(tgt.ra.radian)
	sin_dec_t = np.sin(tgt.dec.radian)
	cos_ra_s  = np.cos(sun_coord.ra.radian)
	cos_dec_s = np.cos(sun_coord.dec.radian)
	sin_ra_s  = np.sin(sun_coord.ra.radian)
	sin_dec_s = np.sin(sun_coord.dec.radian)

	# rotation matrix to convert a vector in celestial coordinates to body-frame coordinates,
	# for the S/C frame aligned to point X towards RA=a, DEC=d, and roll about X of r:
	# for reference only
	#   cosd cosa,					 cosd sina						sind
	#  -cosr sina -sinr sind cosa,	 cosr cosa - sinr sind sina,	sinr cosd
	#   sinr sina - cosr sind cosa,	-sinr cosa - cosr sind sina,	cosr cosd
	#
	# roll is defined by rotating Sun vector into body frame, then minimizing the Z component
	cc = cos_ra_s * cos_dec_s
	sc = sin_ra_s * cos_dec_s
	arg1 = sin_ra_t * cc - cos_ra_t * sc
	arg2 = cos_dec_t * sin_dec_s - sin_dec_t * (cos_ra_t * cc + sin_ra_t * sc)
	phi = np.arctan2(arg1, arg2)

	# there are two solutions - check that resulting vector has positive Z component
	sinr = np.sin(phi)
	cosr = np.cos(phi)

	# Rotate Sun vector into body frame
	# start by confirming that projection onto Z axis is positive - if not we have to change
	# roll by 180 degrees
	z_sun1 = ( sinr*sin_ra_t - cosr*sin_dec_t*cos_ra_t) * cc
	z_sun2 = (-sinr*cos_ra_t - cosr*sin_dec_t*sin_ra_t) * sc
	z_sun3 = cosr*cos_dec_t*sin_dec_s
	z_sun = z_sun1 + z_sun2 + z_sun3
	#if (z_sun < 0.):
	#	phi = phi + np.pi
	phi = np.where(z_sun < 0., phi + np.pi, phi)

	# done with main task of this routine.
 
	# convert to degrees
	nominal_roll = phi * 180./np.pi


	# wrap roll to be between 0 and 360.
	nominal_roll = np.where(nominal_roll >= 0.,nominal_roll, nominal_roll+360.)

	# define position angles (orientation East of North)
	# observatory +Y axis
	pa_obs_y = nominal_roll - 90.
	# Projected WFI focal plane local +X axis:
	pa_fpa_local_x = nominal_roll - 30.
	# Projected WFI focal plane local +Y axis:
	pa_fpa_local_y = nominal_roll - 120.

	# wrap all to be between 0, 360 degrees
	pa_obs_y = np.where(pa_obs_y >= 0.,pa_obs_y, pa_obs_y+360.)
	pa_fpa_local_x = np.where(pa_fpa_local_x >= 0.,pa_fpa_local_x, pa_fpa_local_x+360.)
	pa_fpa_local_y = np.where(pa_fpa_local_y >= 0.,pa_fpa_local_y, pa_fpa_local_y+360.)

    # cross checks
    # check on other projections of Sun vector onto body axes as a sanity check
	x_sun = cos_ra_t*cos_dec_t*cc + cos_dec_t*sin_ra_t*sc + sin_dec_t*sin_dec_s
	sunang_x = np.arccos(x_sun) * 180./np.pi

	sinr = np.sin(phi)
	cosr = np.cos(phi)
	y_sun1 = (-cosr*sin_ra_t - sinr*sin_dec_t*cos_ra_t) * cc
	y_sun2 = ( cosr*cos_ra_t - sinr*sin_dec_t*sin_ra_t) * sc
	y_sun3 = sinr*cos_dec_t*sin_dec_s
	y_sun = y_sun1 + y_sun2 + y_sun3
	sunang_y = np.arccos(y_sun) * 180./np.pi

	z_sun1 = ( sinr*sin_ra_t - cosr*sin_dec_t*cos_ra_t) * cc
	z_sun2 = (-sinr*cos_ra_t - cosr*sin_dec_t*sin_ra_t) * sc
	z_sun3 = cosr*cos_dec_t*sin_dec_s
	z_sun = z_sun1 + z_sun2 + z_sun3
	sunang_z = np.arccos(z_sun) * 180./np.pi

	# generate output 
	# There is likely a more elegant way to print this table, but it will do
	#      ----------   ------    ------    -----
	print("   Date       RA_Sun       DEC_Sun      Roll Pitch_ok  Xang   Yang   Zang",file=ofile)
	for i_d in range (n_samp):
		# get the date string only - hms are all zeros so don't clutter output
		t_str = t_1year[i_d].iso
		d_str = t_str[0:10]
		ra_val = sun_coord.ra[i_d]
		dec_val = sun_coord.dec[i_d]
		roll_val = nominal_roll[i_d]
		p_ok = good_angles[i_d]
		xang = sunang_x[i_d]
		yang = sunang_y[i_d]
		zang = sunang_z[i_d]
		print("{0:s}   {1:6.1f}   {2:6.1f}   {3:6.1f}  {4:2d}    {5:6.1f}  {6:6.1f} {7:6.1f}".format(d_str, ra_val, dec_val, roll_val, p_ok,xang,yang,zang),file=ofile)
		# end of loop over days
		
	print("\n\n",file=ofile)
	# end of loop over test lines of sight
	
ofile.close()

		