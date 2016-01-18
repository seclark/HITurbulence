import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy.fftpack
from scipy import ndimage as ndi

# HI DATA
galfa_data_fn = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/SC_241.66_28.675.best_20.fits"
galfa_data = fits.getdata(galfa_data_fn)
dx = 512
galfa_data = galfa_data[500:500+dx, 4800:4800+dx]
no_fluct_data = galfa_data - np.nanmean(galfa_data)

#no_fluct_data = no_fluct_data[500:500+dx, 4800:4800+dx]

# B-field data
QU = fits.getdata("/Volumes/DataDavy/Planck/SC_241.66_28.675.allBWmustdie.PlanckQU_Equi_IAU.fits")
Q = QU[0, :, :].T 
U = QU[1, :, :].T 
theta_353 = np.mod(0.5*np.arctan2(-U, -Q), np.pi)
theta_353_chunk = theta_353[500:500+dx, 4800:4800+dx]

def cut_out_SF(data, x0, y0, rad, mirror = True, square = False):
    # Cut out an annulus
    mnvals = np.indices(data.shape)
    mvals = mnvals[:, :][0] # These are the y points
    nvals = mnvals[:, :][1] # These are the x points
    rads = np.zeros(data.shape, np.float_)
    rads = np.sqrt((mvals-y0)**2 + (nvals-x0)**2)
    
    if square == False:
        data[rads >= rad] = 0
    data = data[y0-rad:y0+rad+1, x0-rad:x0+rad+1]
    
    data = data - np.nanmean(data)
    
    FT = scipy.fftpack.fft2(data)
    AC = (scipy.fftpack.ifft2(FT * np.conjugate(FT))).real
    SF = 2*(np.mean(data**2)-AC)
    
    if mirror == True:
        SF = np.roll(np.roll(SF, 256, axis=1), 256,axis=0)
    
    return data, SF
    
rad = 150
x0 = 300
y0 = 300
#SF_chunk = galfaSF[x0-rad:x0+rad, y0-rad:y0+rad]

data, cc_sf = cut_out_SF(no_fluct_data, x0, y0, rad, square = True)
datay, datax = data.shape
print(datay, datax)

rollcc = np.roll(np.roll(cc_sf, datax/2, axis=1), datax/2, axis=0)
rotrollcc = ndi.interpolation.rotate(rollcc, np.degrees(theta_353_chunk[x0, y0]), reshape=False)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.imshow(data)
ax2.imshow(cc_sf)
ax3.imshow(rotrollcc)

ax4.loglog(rotrollcc[datax/2, datax/2:] - np.nanmin(rotrollcc), ".", color = "red")
ax4.loglog(rotrollcc[datax/2:, datax/2] - np.nanmin(rotrollcc), ".", color = "blue")
ax4.set_xlim(10, rad)

"""
radius = 150
ny, nx = no_fluct_data.shape
rotrollcc = np.zeros((dx, dx), np.float_)
count = 0
for x0 in xrange(nx):
    for y0 in xrange(ny):
        if (x0 > radius) and (y0 > radius) and (x0 < (nx - radius)) and (y0 < (ny - radius)):
            local_data, local_sf = cut_out_SF(no_fluct_data, x0, y0, radius)
            rotrollcc += ndi.interpolation.rotate(local_sf, np.degrees(theta_353_chunk[y0, x0]), reshape=False)
            count += 1
            
rotrollcc_corner = rotrollcc[256:, 256:]
plt.imshow(rotrollcc_corner/count)
"""
