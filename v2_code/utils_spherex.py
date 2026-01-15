#!/usr/bin/env python3

import os, glob, sys
import numpy as np
import warnings
from enum import IntFlag, auto

import urllib.request
import urllib.error

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.nddata import Cutout2D
from astropy.table import Table, QTable
from astropy import units as u
from astropy.stats import sigma_clipped_stats
import astropy.constants as const

# Astroquery for IRSA access
from astroquery.ipac.irsa import Irsa

# Reproject for image alignment
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from astropy.utils.console import ProgressBar

# Could move ths into the file
from utils_z0mgs_images import deproject

# Used to analyze the cube
from scipy.interpolate import make_smoothing_spline
from spectral_cube import SpectralCube

#import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
#import astropy.utils.exceptions
#warnings.simplefilter('ignore', category=astropy.utils.exceptions.AstropyWarning)                     

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines to query data from IRSA
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def check_spherex_collections():
    """Query IRSA's collection list to see which collections hold spherex
    data. Use this to inform the image search.
    """
    
    collections = Irsa.list_collections(servicetype='SIA', filter='spherex')

    print(f"Found {len(collections)} SPHEREX collections:")
    for collection in collections['collection']:
        print(f"  - {collection}")

    return()


def search_spherex_images(
        target: str = None,
        coordinates: SkyCoord = None,
        radius: u.Quantity = 10*u.arcmin,
        collection: str = 'spherex_qr2',
        verbose: bool = True) -> Table:
    
    """
    From Adam Ginsburg (pared down a lot)

    Search for SPHEREX images in IRSA archive.

    Parameters
    ----------
        
    target : str, optional
    Target name (e.g., 'M31', 'NGC 1234')
    
    coordinates : SkyCoord, optional
    Target coordinates

    radius : Quantity
    Search radius

    collection : str (default 'spherex_qr2')
    Collection holding the images, will need to be managed    

    Returns
    -------
    Table
    
    Table of available SPHEREX images
    """
    
    if verbose:
        print(f"Searching for SPHEREX images...")

    # Make sure we have target coordinates
    if coordinates is None and target is not None:
        coordinates = SkyCoord.from_name(target)
    elif coordinates is None:
        raise ValueError("Either target name or coordinates must be provided")

    if verbose:
        print(f"Target coordinates: {coordinates.to_string('hmsdms')}")
        print(f"Search radius: {radius}")

    # Search for images in the specified collection
    try:
        images = Irsa.query_sia(pos=(coordinates, radius), collection=collection)
    except Exception as e:
        if verbose:
            print(f"SIA query failed: {e}")

    if verbose:
        print(f"Found {len(images)} images")
        if len(images) > 0:
            print("Available columns (first 10): ", images.colnames[:10])  # Show first 10 columns

    return(images)


def download_images(
        images: Table,
        max_images: int = None,
        outdir: str = '',
        alt_dirs: list = [str],
        incremental: bool = True,
        verbose: bool = True) -> list[str]:
    """Download SPHEREX images from a table produced by the IRSA query above.

    Modified from an original version by Adam Ginsburg.
    
    Parameters
    ----------
    
    images : Table
             Table of images from search_spherex_images
        
    max_images : int, optional
                 Maximum number of images to download

    outdir : str, optional
             Location where the download directs

    alt_dirs : list of strings, optional
             Alternative directories where the file might be

    incremental : bool, default True
                  Check if file is present before download

    verbose : bool, default True
              Turn on extra print statements
    
    Returns
    -------
    
    list[str] : List of downloaded file paths

    """

    # If an empty table, return
    if len(images) == 0:
        if verbose:
            print("No images to download")
        return([])

    # If we're capped in files to download impose that
    if max_images is not None:
        images = images[:max_images]

    # Initialize record
    downloaded_files = []

    # Loop and download
    if verbose:
        print(f"Downloading {len(images)} images...")
        if len(images) > 10:
            print("  (This may take a while for large datasets...)")

    # Loop over the list of images
    for ii, this_row in enumerate(images):

        # Determine the access URL        
        try:
            this_url = None
            # ... try the different potential column names
            for url_col in ['access_url', 'cloud_access', 'download_url', 'url']:
                if url_col in this_row.colnames and this_row[url_col]:
                    this_url = str(this_row[url_col])
                    break

            # ... catch the error case
            if this_url is None:
                if verbose:
                    print(f"  Skipping image {ii+1}: No access URL found")
                continue

            # Generate filename

            # AKL: Note that you cannot just use the obsid because two
            # images with different filters are obtained with the same
            # obsid - parse the full file name.

            # Observation ID + detector should be unique, but
            # processing time in the ID stamp is also potentially of
            # interest.
           
            #obs_id = this_row.get('obs_id', f'spherex_{ii:04d}')
            
            obs_fname = this_url.split('/')[-1]            
            this_fname = outdir + obs_fname

            # Check if the filename is present already. If incremental
            # is set to True proceed with a notification. Else delete
            # the file.
                
            if os.path.isfile(this_fname):
                if incremental:
                    if verbose:
                        print(f"  Image {ii+1}/{len(images)}: {this_fname} already exists.")
                        print(f"  Incremental is set to TRUE.")
                        downloaded_files.append(str(this_fname))
                    continue
                else:
                    if verbose:
                        print(f"  Image {ii+1}/{len(images)}: {this_fname} already exists.")
                        print(f"  Incremental is set to FALSE. Will proceed and overwrite.")
                    os.system('rm -rf '+this_fname)
            else:

                # If the file is not found but some alternate
                # locations are supplied and the user has set
                # "incremental" then search those (using glob to allow
                # wildcards) and see if the file is there. If so copy
                # it.
                
                if (len(alt_dirs) > 0) and incremental:
                    found_file_elsewhere = False
                    for this_alt_dir in alt_dirs:
                        if found_file_elsewhere:
                            continue
                        
                        alt_flist = glob.glob(this_alt_dir+obs_fname)                        
                        if len(alt_flist) > 0:
                            found_file_elsewhere = True
                            print(f"  Image {ii+1}/{len(images)}: {this_fname} already exists but in another directory.")
                            print(f"  Incremental is set to TRUE.")
                            print(f"  Copying the file.")
                            os.system('cp '+this_alt_dir+obs_fname+' '+this_fname)
                            downloaded_files.append(str(this_fname))

            # Download the file
            if verbose:
                print(f"  Downloading image {ii+1}/{len(images)}: {this_fname}")

            # Download using urllib first, then open with astropy
            try:                
                # Download to a temporary location first
                temp_fname = this_fname + '.tmp'
                urllib.request.urlretrieve(this_url, temp_fname)
            
                # Verify it's a valid FITS file by opening it
                with fits.open(temp_fname) as this_hdu:                
                    # Save as final file
                    this_hdu.writeto(this_fname, overwrite=True)

                # Remove temporary file
                os.system('rm -rf '+temp_fname)

            except urllib.error.URLError as e:
                if verbose:
                    print(f"    URL error: {e}")
                continue            
            except Exception as fits_error:            
                # Clean up temp file if it exists
                temp_fname = this_fname + '.tmp'
                if os.path.isfile(temp_fname):
                    os.system('rm -rf '+temp_fname)
                    # Will crash the program?
                raise fits_error

            # If we made it here we're successful
            downloaded_files.append(str(this_fname))
        
        except Exception as e:
            if verbose:
                print(f"  Failed to download image {ii+1}: {e}")
            continue

    if verbose:
        print(f"Successfully downloaded {len(downloaded_files)} images")

    return(downloaded_files)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines to support building a cube
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def make_wavelength_image(
        hdu_list = None,
        use_hdu = None,
):
    """Use astropy WCS to construct images of wavelength and bandwidth
    across a SPHEREx image.

    Feed in the HDUlist for an image and the HDU holding the image of
    interest. Program returns 

    lam, bw

    The central wavelength and bandwidth of the image in microns.

    Based on IRSA tutorial.
    """

    # Testing shows that the various image extensions do produce the
    # same output images.
    
    this_header = hdu_list[use_hdu].header.copy()
    this_shape = hdu_list[use_hdu].data.shape

    # Remove SIP coefficients. Mostly this just avoids an annoying
    # error message, since setting spectral_wcs.sip = None below turns
    # off their application anyways.
    
    keywords_to_remove = ['A_?_?', 'B_?_?', 'AP_?_?', 'BP_?_?',
                          '*_ORDER']
    for this_pattern in keywords_to_remove:
        for this_keyword in this_header.copy():
            if glob.fnmatch.fnmatch(this_keyword, this_pattern):
                del this_header[this_keyword]
    
    # Key call. Feed the header associated with the desired image,
    # also pass the parent HDU list to handle distortions/wavelength
    # lookup, and use the "W"avelength transform.

    # This all relies on astropy being wired under the hood to use the
    # WCS-WAVE table in the final extension.
    
    spectral_wcs = WCS(this_header, fobj=hdu_list, key="W")

    # Turn off spatial distortion terms
    spectral_wcs.sip = None

    # Create a grid of pixel coords
    yy, xx = np.indices(this_shape)
    
    # Evaluate the WCS to get the wavelength and bandwidth
    lam, bw = spectral_wcs.pixel_to_world(xx, yy)

    # Get the units (not used right now)
    lam_unit = spectral_wcs.wcs.cunit[0]
    bw_unit = spectral_wcs.wcs.cunit[1]

    # Return images of lambda and bandwidth
    return((lam, bw))

def make_cube_header(
        center_coord,
        pix_scale = 6. / 3600.,
        extent = None, extent_x = None, extent_y = None,
        nx = None, ny = None,
        lam_min = 0.7, lam_max = 5.2, lam_step = 0.02,
        lam_unit = 'um',
        return_header=False):
    """Make a 2D FITS header centered on the coordinate of interest with a
    user-specififed pixel scale and extent and wavelength axis.

    
    Parameters
    ----------

    center_coord : `~astropy.coordinates.SkyCoord` object or
        array-like Sky coordinates of the image center. If array-like
        then (ra, dec) in decimal degrees assumes.

    pix_scale : required. Size in decimal degrees of a pixel. Can be
        an array in which case it is pixel scale along x and y (e.g.,
        as returned by proj_pixel_scales).

    extent : the angular extent of the image in decimal degrees for both x and y.

    extent_x : the angular extent of the image along the x coordinate

    extent_y : the angular extent of the image along the y coordinate

    nx : the number of x pixels (not needed with extent_x and pix_scale)

    ny : the number of y pixels (not needed with extent_y and pix_scale)

    lam_min, lam_max, lam_step : minimum, maximum, and channel width
        for wavelength axis in microns

    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    
    # Figure out the center
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if isinstance(center_coord, str):
        coordinates = SkyCoord.from_name(center_coord)
        ra_ctr = coordinates.ra.degree
        dec_ctr = coordinates.dec.degree
    elif isinstance(center_coord, SkyCoord):
        ra_ctr = center_coord.ra.degree
        dec_ctr = center_coord.dec.degree
    else:
        ra_ctr, dec_ctr = center_coord
        if hasattr(ra_ctr, 'unit'):
            ra_ctr = ra_ctr.to(u.deg).value
            dec_ctr = dec_ctr.to(u.deg).value

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Figure out the pixel scale
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    if pix_scale is None:
        print("Pixel scale not specified. Returning.")
        return()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    
    # Figure out image extent
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if extent is not None:
        extent_x = extent
        extent_y = extent
    
    if (nx is not None) and (ny is not None):
        if isinstance(pix_scale, np.ndarray):
            extent_x = pix_scale[0] * nx
            extent_y = pix_scale[1] * ny
        else:
            extent_x = pix_scale * nx
            extent_y = pix_scale * ny
    elif (extent_x is not None) and (extent_y is not None):
        if isinstance(pix_scale, np.ndarray):
            nx = int(np.ceil(extent_x*0.5 / pix_scale[0]) * 2 + 1)
            ny = int(np.ceil(extent_y*0.5 / pix_scale[1]) * 2 + 1)
        else:
            nx = int(np.ceil(extent_x*0.5 / pix_scale) * 2 + 1)
            ny = int(np.ceil(extent_y*0.5 / pix_scale) * 2 + 1)            
    else:
        print("Extent not specified. Returning.")
        return()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Make the wavelength axis
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    lam_array = np.arange(lam_min, lam_max + lam_step, lam_step)
    nz = len(lam_array)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Make the FITS header
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    
    
    hdu = fits.PrimaryHDU()    
    hdu.header = fits.Header()
    hdu.header['NAXIS'] = 3

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Spatial information
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    hdu.header['NAXIS1'] = nx
    hdu.header['NAXIS2'] = ny
    
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CRVAL1'] = ra_ctr
    hdu.header['CRPIX1'] = np.float16((nx / 2) * 1 - 0.5)

    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL2'] = dec_ctr
    hdu.header['CRPIX2'] = np.float16((ny / 2) * 1 - 0.5)
    
    if isinstance(pix_scale, np.ndarray):    
        hdu.header['CDELT1'] = -1.0 * pix_scale[0]
        hdu.header['CDELT2'] = 1.0 * pix_scale[1]
    else:
        hdu.header['CDELT1'] = -1.0 * pix_scale
        hdu.header['CDELT2'] = 1.0 * pix_scale
            
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['RADESYS'] = 'FK5'
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Spectral information
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    hdu.header['NAXIS3'] = nz
    hdu.header['CTYPE3'] = 'WAVE'
    hdu.header['CUNIT3'] = 'um'
    hdu.header['CRPIX3'] = 1
    hdu.header['CRVAL3'] = lam_array[0]
    hdu.header['CDELT3'] = lam_step

    # Return header or HDU    
    if return_header:
        return(hdu.header)
    else:
        return(hdu)

# Potentially useful but interface with numpy is not straightforward.

class SpherexFlag(IntFlag):
    TRANSIENT = auto()
    OVERFLOW = auto()
    SUR_ERROR = auto()
    BLANK_1 = auto()
    PHANTOM = auto()
    REFERENCE = auto()
    NONFUNC = auto()
    DICHROIC = auto()
    BLANK_2 = auto()
    MISSING_DATA = auto()
    HOT = auto()
    COLD = auto()    
    FULLSAMPLE = auto()
    BLANK_3 = auto()
    PHANMISS = auto()
    NONLINEAR = auto()
    BLANK_4 = auto()    
    PERSIST = auto()
    BLANK_5 = auto()        
    OUTLIER = auto()
    BLANK_6 = auto()        
    SOURCE = auto()

def spherex_flag_dict():
    """
    """

    # From the header, flag definitions
    
    #HIERARCH MP_TRANSIENT = 0 / Transient detected during SUR                       
    #HIERARCH MP_OVERFLOW = 1 / Overflow detected during SUR                         
    #HIERARCH MP_SUR_ERROR = 2 / Error in onboard processing                         
    #HIERARCH MP_PHANTOM = 4 / Phantom pixel                                         
    #HIERARCH MP_REFERENCE = 5 / Reference pixel                                     
    #HIERARCH MP_NONFUNC = 6 / Permanently unusable                                  
    #HIERARCH MP_DICHROIC = 7 / Low efficiency due to dichroic                       
    #HIERARCH MP_MISSING_DATA = 9 / Onboard data lost                                
    #MP_HOT  =                   10 / Hot pixel                                      
    #MP_COLD =                   11 / Anomalously low signal                         
    #HIERARCH MP_FULLSAMPLE = 12 / Pixel full sample history is available            
    #HIERARCH MP_PHANMISS = 14 / Phantom correction was not applied                  
    #HIERARCH MP_NONLINEAR = 15 / Nonlinearity correction cannot be applied reliably 
    #HIERARCH MP_PERSIST = 17 / Persistent charge above threshold                    
    #HIERARCH MP_OUTLIER = 19 / Pixel flagged by Detect Outliers                     
    #HIERARCH MP_SOURCE = 21 / Pixel mapped to a known source    
    
    this_dict = {
        'TRANSIENT' : 0,
        'OVERFLOW' : 1,
        'SUR_ERROR' : 2,
        'PHANTOM' : 4,
        'REFERENCE' : 5,
        'NONFUNC' : 6,
        'DICHROIC' : 7,
        'MISSING_DATA' : 9,
        'HOT' : 10,
        'COLD' : 11,
        'FULLSAMPLE' : 12,
        'PHANMISS' : 14,
        'NONLINEAR' : 15,
        'PERSIST' : 17,
        'OUTLIER' : 19,
        'SOURCE' : 21,        
    }
    
    return(this_dict)

def estimate_spherex_bkgrd(
        image = None,
        header = None,
        lam = None,
        bw = None,
        gal_coord = None,
        gal_rad_deg = 10./60.,
        gal_incl = 0.0,
        gal_pa = 0.0,
        frac_bw_step = 0.25,
        ):
    """
    Estimate a wavelength dependent background for one image.
    """

    # Initialize
    bkgrd = np.zeros_like(image)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Mask out the galaxy
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    # Set the center to the image center if not supplied
    if gal_coord is None:
        ny, nx = image.shape
        wcs = WCS(header)
        gal_coord = pixel_to_skycoord(nx / 2, ny / 2, wcs, origin=0)
        
    # Calculate the galaxy footprint
    radius_deg, projang_deg = \
        deproject(
            center_coord=gal_coord, incl=gal_incl, pa=gal_pa,
            template_header = header,
            return_offset = False)
    
    # Create a copy with the galaxy masked
    temp_data = image.copy()
    gal_ind = np.where(radius_deg < gal_rad_deg)
    temp_data[gal_ind] = np.nan

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Fit a median at each wavelength
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            
    med_bw = np.nanmedian(bw.to(u.um)).value
    min_lam = np.nanmin(lam).value
    max_lam = np.nanmax(lam).value

    step = med_bw * frac_bw_step
    
    lams_for_bkgrd = np.arange(min_lam-med_bw,
                               max_lam+med_bw, step)

    for this_lam in lams_for_bkgrd:
        
        this_ind = np.where(np.abs(lam.value - this_lam) <= 0.5*step)
        if len(this_ind) == 0:
            continue
        this_bkgrd_val = np.nanmedian(temp_data[this_ind])
        bkgrd[this_ind] = this_bkgrd_val

    return(bkgrd)

def bksub_images(
        image_list = None,
        indir_ext = 'raw/',
        outdir_ext = 'bksub/',
        # These parameters determin what mask to apply to avoid
        # subtracting the galaxy
        gal_coord = None,
        gal_rad_deg = 10./60.,
        gal_incl = 0.0,
        gal_pa = 0.0,
        frac_bw_step = 0.25,        
        sub_zodi = True,
        flags_to_use = ['SUR_ERROR','NONFUNC','MISSING_DATA',
                       'HOT','COLD','NONLINEAR','PERSIST'],
        overwrite = True):
    """Background subtract a set of SPHEREx images.

    image_list: list of images to process

    indir_ext ( 'raw/' ) : images are assumed to have indir_ext in the
    file path, which is swapped to outdir_ext for output

    outdir_ext ( 'bksub/' ) : see above

    sub_zodi ( True ) : subtract zodiacal light before doing anything else

    frac_bw_step ( 0.25 ) : fraction of the typical bandwidth to step
    by when dividing the image into wavelength bins

    Parameters to define a simple mask for the location of the galaxy
    to avoid including it in the background estimate.

    gal_coord ( None ) : if None takes image center

    gal_rad_deg ( 10./60. ) : radius in degrees
    
    gal_incl ( 0.0 ) : inclination in degrees
    gal_pa ( 0.0 ) : position angle in degrees

    """

    # Loop over the provided image list

    bksub_image_list = []
    
    for this_fname in ProgressBar(image_list):

        this_hdu_list = fits.open(this_fname)

        # Separate the file name and check if the file already exists
        
        outname_bksub = this_fname.replace(indir_ext+'level2',
                                           outdir_ext+'bksub_level2')
        outname_bkgrd = this_fname.replace(indir_ext+'level2',
                                           outdir_ext+'bkgrd_level2')

        if os.path.isfile(outname_bksub) and overwrite == False:
            bksub_image_list.append(outname_bksub)
            continue

        # Make an image of wavelength and bandwidth
        
        lam, bw = make_wavelength_image(
            hdu_list = this_hdu_list,
            use_hdu = 'IMAGE',
        )

        # Implement flags

        hdu_flags = this_hdu_list['FLAGS']        
        mask = make_mask_from_flags(
            hdu_flags.data,
            flags_to_use = flags_to_use,
        )

        hdu_image = this_hdu_list['IMAGE']
        image_header = hdu_image.header
        masked_data = hdu_image.data.copy()
        masked_data[mask] = np.nan

        # Subtract the zodiacal light if desired
        
        if sub_zodi:
            hdu_zodi = this_hdu_list['ZODI']            
            masked_data = masked_data - hdu_zodi.data

        # Fit a background (or leave it at 0.0)
        
        bkgrd = estimate_spherex_bkgrd(
            image = masked_data,
            header = image_header,
            lam = lam,
            bw = bw,
            gal_coord = gal_coord,
            gal_rad_deg = gal_rad_deg,
            gal_incl = gal_incl,
            gal_pa = gal_pa,
            frac_bw_step = frac_bw_step,
        )

        # Subtract the background
        
        masked_data -= bkgrd

        # Save in FITS HDUs
        
        hdu_masked_image = fits.ImageHDU(data=masked_data, header=image_header,
                                        name='BKSUB')
        hdu_bkgrd_image = fits.ImageHDU(data=bkgrd, header=image_header,
                                        name='BKGRD')
        this_hdu_list.insert(0,hdu_masked_image)
        this_hdu_list.append(hdu_bkgrd_image)
        
        # Write out to images
        
        this_hdu_list.writeto(outname_bksub, overwrite=overwrite)
        
    return(bksub_image_list)


def make_mask_from_flags(
        flag_image,
        flags_to_use = ['SUR_ERROR','NONFUNC','MISSING_DATA',
                        'HOT','COLD','NONLINEAR','PERSIST'],        
):
    """Given a list of flag names to use construct a mask from the input
    flag image. The flag image is a bitmask, and the mapping between
    flags and bits is defined in the explanatory supplement and coded
    into a dictionary above.

    THIS COULD ALL USE CHECKING!!!

    """

    # From the explanatory supplement

    # Suggested flags for background estimation masking:

    # OVERFLOW
    # SUR_ERROR
    # NONFUNC
    # MISSING_DATA
    # HOT
    # COLD
    # NONLINEAR
    # PERSIST
    # OUTLIER
    # SOURCE
    # TRANSIENT

    # Suggested flags for on source photometry:

    # SUR_ERROR
    # NONFUNC
    # MISSING_DATA
    # HOT
    # COLD
    # NONLINEAR
    # PERSIST

    use_flag_ind = []
    flag_dict = spherex_flag_dict()
    for this_flag in flags_to_use:
        try:
            this_ind = flag_dict[this_flag.upper()]
        except KeyError:
            print("Flag unrecognized: ", this_flag)
            continue
        use_flag_ind.append(this_ind)
    
    n_flags = 22

    # Make an array of powers of 2 to cover each relevant bit
    powers_of_2 = 1 << np.arange(n_flags)

    # Copy the flag image to AND against each flag
    flag_cube = np.repeat(flag_image[:,:,np.newaxis], n_flags, axis=2)
    
    # Use bitwise AND to hash against each flag
    mask_cube = (flag_cube & powers_of_2) != 0
    
    # Initialize the image mask
    mask = np.zeros_like(flag_image, dtype=bool)

    # Loop over and accumulate the requested flags
    for ii in np.arange(n_flags):

        if ii not in use_flag_ind:
            continue

        mask = mask | (mask_cube[:,:,ii])
        
    return(mask)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routine to extract an SED
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def extract_spherex_sed(
        target_coord,        
        image_list = [],
        outfile = None,
        overwrite=True):
    """Extract the SED at a target position (target_coord) from a list of
    images. Write it to an output file (as a text table).

    There is no gridding here, the SED will just record the wavelength
    and bandwidth from each image in the list at that location. Useful
    to directly see the data. The table also includes the image file
    name.

    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Get the coordinates
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    if isinstance(target_coord, str):
        coordinates = SkyCoord.from_name(target_coord)
        ra_deg = coordinates.ra.degree
        dec_deg = coordinates.dec.degree
    elif isinstance(target_coord, SkyCoord):
        ra_deg = target_coord.ra.degree
        dec_deg = target_coord.dec.degree
    else:
        ra_deg, dec_deg = target_coord
        if hasattr(ra_deg, 'unit'):
            ra_deg = ra_deg.to(u.deg).value
            dec_deg = dec_deg.to(u.deg).value

    target_coord = SkyCoord(
        ra = ra_deg * u.deg, dec=dec_deg* u.deg,
        frame = 'icrs')
                                        
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Loop over the image list
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    val_ra = np.zeros(len(image_list))*np.nan
    zodi_ra = np.zeros(len(image_list))*np.nan
    lam_ra = np.zeros(len(image_list))*np.nan
    bw_ra = np.zeros(len(image_list))*np.nan
    file_list = []
    
    counter = 0
    
    for this_fname in ProgressBar(image_list):        
        
        this_hdu_list = fits.open(this_fname)
        hdu_image = this_hdu_list['IMAGE']
        hdu_zodi = this_hdu_list['ZODI']        
        image_header = hdu_image.header
        this_wcs = WCS(image_header)
        
        lam, bw = make_wavelength_image(
            hdu_list = this_hdu_list,
            use_hdu = 'IMAGE',
        )

        y_pix, x_pix = this_wcs.world_to_array_index(target_coord)
        this_shape = hdu_image.data.shape
        if (y_pix < 0) | (y_pix >= this_shape[0]) | \
           (x_pix < 0) | (x_pix >= this_shape[1]):
            continue
        
        this_val = hdu_image.data[y_pix, x_pix]
        this_zodi = hdu_zodi.data[y_pix, x_pix]        
        this_lam = lam.data[y_pix, x_pix]
        this_bw = bw.data[y_pix, x_pix]

        val_ra[counter] = this_val
        zodi_ra[counter] = this_zodi
        lam_ra[counter] = this_lam
        bw_ra[counter] = this_bw
        file_list.append(this_fname.split('/')[-1])
        
        counter += 1

    flist_ra = np.array(file_list)
        
    keep_ind = np.where(np.isfinite(val_ra))
    val_ra = val_ra[keep_ind]
    zodi_ra = zodi_ra[keep_ind]
    lam_ra = lam_ra[keep_ind]
    bw_ra = bw_ra[keep_ind]
    flist_ra = flist_ra[keep_ind]
        
    tab = QTable([lam_ra*u.um, bw_ra*u.um, val_ra*u.MJy/u.sr, zodi_ra*u.MJy/u.sr,
                  flist_ra],
                 names=['lam','bw','val','zodi','fname'])

    if outfile is not None:
        tab.write(outfile, overwrite=overwrite, format='ascii.ecsv')
    
    return(tab)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines to cubes
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def build_sed_cube(
        target_hdu = None,
        image_list = [],
        ext_to_use = 'IMAGE',
        outfile = None,
        overwrite=True):
    """Build cubes of wavelength, bandwidth, and intensity (without any
    gridding) for a stack of SPHEREx images. By loading all three,
    this produces the measured SED at each location in the cube.

    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialize the output
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    target_header = target_hdu.header
    nx = target_header['NAXIS1']
    ny = target_header['NAXIS2']
    nz = target_header['NAXIS3']

    target_header_2d = target_header.copy()
    target_header_2d['NAXIS'] = 2
    del target_header_2d['NAXIS3']
    del target_header_2d['CRVAL3']
    del target_header_2d['CDELT3']
    del target_header_2d['CRPIX3']
    del target_header_2d['CTYPE3']
    del target_header_2d['CUNIT3']
    
    lam_cube = np.zeros((nz,ny,nx),dtype=np.float32)*np.nan
    bw_cube = np.zeros((nz,ny,nx),dtype=np.float32)*np.nan
    int_cube = np.zeros((nz,ny,nx),dtype=np.float32)*np.nan

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Loop over images
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    zz = 0
    for this_fname in ProgressBar(image_list):   

        # Open this file
        
        this_hdu_list = fits.open(this_fname)
        hdu_image = this_hdu_list[ext_to_use]
        image_header = hdu_image.header

        # Calculate wavelength and bandwidth per pixel
        
        lam, bw = make_wavelength_image(
            hdu_list = this_hdu_list,
            use_hdu = ext_to_use,
        )

        # Make FITS HDUs (in memory) out of the wavelength and bandwidth
        
        hdu_lam = fits.PrimaryHDU(lam, image_header)
        hdu_bw = fits.PrimaryHDU(bw, image_header)

        # Reproject image, wavelength, and bandwidth to target header
        
        missing = np.nan
        
        reprojected_image, footprint_image = \
            reproject_interp(hdu_image, target_header_2d, order='bilinear')
        reprojected_image[footprint_image == 0] = missing

        reprojected_lam, footprint_lam = \
            reproject_interp(hdu_lam, target_header_2d, order='bilinear')
        reprojected_lam[footprint_lam == 0] = missing

        reprojected_bw, footprint_bw = \
            reproject_interp(hdu_bw, target_header_2d, order='bilinear')
        reprojected_bw[footprint_bw == 0] = missing                

        # Record these into the cubes

        lam_cube[zz,:,:] = reprojected_lam
        bw_cube[zz,:,:] = reprojected_bw
        int_cube[zz,:,:] = reprojected_image
        
        zz += 1

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Write to disk
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    int_header = target_header.copy()
    int_header['BUNIT'] = 'MJy / sr'
    int_hdu = fits.PrimaryHDU(int_cube, int_header)
    if outfile is not None:
        int_hdu.writeto(outfile, overwrite=overwrite)

    lam_header = target_header.copy()
    lam_header['BUNIT'] = 'um'
    lam_hdu = fits.PrimaryHDU(lam_cube, lam_header)
    if outfile is not None:
        lam_hdu.writeto(outfile.replace('.fits','_lam.fits')
                        , overwrite=overwrite)
        
    bw_header = target_header.copy()
    bw_header['BUNIT'] = 'um'
    bw_hdu = fits.PrimaryHDU(bw_cube, bw_header)
    if outfile is not None:
        bw_hdu.writeto(outfile.replace('.fits','_bw.fits')
                        , overwrite=overwrite)
        
    return((int_hdu, lam_hdu, bw_hdu))

def grid_spherex_cube(
        # Input cubes
        int_cube = None,
        lam_cube = None,
        bw_cube = None,
        # Desired wavelength grid
        lam_min = 0.7, lam_max = 5.2, lam_step = 0.02,
        lam_unit = 'um',
        # Method
        method = 'TOPHAT',
        outfile = None,
        overwrite=True):
    """Grid an SED cube like the one produced by BUILD_SED_CUBE into a
    regular cube with a grid of wavelengths.

    """

    if lam_cube is None:
        lam_cube = int_cube.replace('.fits','_lam.fits')        
        
    if bw_cube is None:
        bw_cube = int_cube.replace('.fits','_bw.fits')

    int_hdu = fits.open(int_cube)[0]
    lam_hdu = fits.open(lam_cube)[0]
    bw_hdu = fits.open(bw_cube)[0]

    int_cube = int_hdu.data
    lam_cube = lam_hdu.data
    bw_cube = bw_hdu.data    
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialize header and output
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    target_header = int_hdu.header.copy()
    nx = target_header['NAXIS1']
    ny = target_header['NAXIS2']

    # Hack to desired new wavelength axis    
    lam_array = np.arange(lam_min, lam_max + lam_step, lam_step)
    nz = len(lam_array)
    target_header['NAXIS'] = 3
    
    target_header['NAXIS3'] = nz
    target_header['CTYPE3'] = 'WAVE'
    target_header['CUNIT3'] = lam_unit
    target_header['CRPIX3'] = 1
    target_header['CRVAL3'] = lam_array[0]
    target_header['CDELT3'] = lam_step

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Loop over channels or pixels
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if method == 'TOPHAT':

        sum_cube = np.zeros((nz,ny,nx),dtype=np.float32)
        bw_sum_cube = np.zeros((nz,ny,nx),dtype=np.float32)
        weight_cube = np.zeros((nz,ny,nx),dtype=np.float32)
        
        for zz in ProgressBar(range(nz)):

            this_lam = lam_array[zz]
            
            # Compare the wavelength at each pixel to the center of
            # the current channel in units of bandwidth
            
            delta = np.abs(this_lam - lam_cube)/bw_cube
            
            # Keep the pixels where that difference is less than half
            # the bandwidth
            
            sed_ind, y_ind, x_ind = \
                np.where(delta <= 0.5)
            
            z_ind = np.zeros_like(y_ind,dtype=int)+zz
            
            sum_cube[z_ind, y_ind, x_ind] = \
                sum_cube[z_ind, y_ind, x_ind] + \
                int_cube[zz, y_ind, x_ind]*1.0

            bw_sum_cube[z_ind, y_ind, x_ind] = \
                 bw_sum_cube[z_ind, y_ind, x_ind] + \
                 bw_cube[zz, y_ind, x_ind]*1.0
            
            weight_cube[z_ind, y_ind, x_ind] = \
                  weight_cube[zz, y_ind, x_ind] + 1.0

        # Calculate weighted average
            
        cube = sum_cube / weight_cube        
        cube[np.where(weight_cube == 0.0)] = np.nan

        bw_cube = bw_sum_cube / weight_cube
        bw_cube[np.where(weight_cube == 0.0)] = np.nan
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Write to disk
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    cube_hdu = fits.PrimaryHDU(cube, target_header)
    if outfile is not None:
        cube_hdu.writeto(outfile, overwrite=overwrite)

    weight_hdu = fits.PrimaryHDU(weight_cube, target_header)
    if outfile is not None:
        weight_hdu.writeto(outfile.replace('.fits','_weight.fits')
                           , overwrite=overwrite)
        
    bw_header = target_header.copy()
    bw_header['BUNIT'] = 'um'
    
    bw_hdu = fits.PrimaryHDU(bw_cube, bw_header)
    if outfile is not None:
        bw_hdu.writeto(outfile.replace('.fits','_bw.fits')
                       , overwrite=overwrite)
    
    return(cube_hdu)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routine to make a line map
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def spherex_line_image(
        target_hdu = None,
        central_lam = 1.87,
        vel_width = 500.,
        frac_thresh = 0.75,
        image_list = [],
        continuum = None,
        operation = 'integrate',
        flags_to_use = ['SUR_ERROR','NONFUNC','MISSING_DATA',
                        'HOT','COLD','NONLINEAR','PERSIST'],
        outfile = None,
        overwrite = True):
    """
    Grid images into a Spherex line-integrated image.
    """

    sol_kms = 2.99792E5
    sol_cgs = 2.99792468E10
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialize the output
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    target_header = target_hdu.header
    nx = target_header['NAXIS1']
    ny = target_header['NAXIS2']

    target_header_2d = target_header.copy()
    target_header_2d['NAXIS'] = 2
    del target_header_2d['NAXIS3']
    del target_header_2d['CRVAL3']
    del target_header_2d['CDELT3']
    del target_header_2d['CRPIX3']
    del target_header_2d['CTYPE3']
    del target_header_2d['CUNIT3']
       
    sum_image = np.zeros((ny,nx),dtype=np.float32)
    weight_image = np.zeros((ny,nx),dtype=np.float32)

    if continuum is not None:
        cont_hdu_list = fits.open(continuum)
        cont_hdu = cont_hdu_list[0]

        missing = np.nan
        reprojected_cont, footprint_cont = \
            reproject_interp(cont_hdu, target_header_2d, order='bilinear')
        reprojected_cont[footprint_cont == 0] = missing
    
    min_lam = central_lam - vel_width/sol_kms*central_lam*0.5
    max_lam = central_lam + vel_width/sol_kms*central_lam*0.5

    # Initialize

    sum_image = np.zeros((ny,nx),dtype=np.float32)
    weight_image = np.zeros((ny,nx),dtype=np.float32)

    bkgrd_sum_image = np.zeros((ny,nx),dtype=np.float32)
    bkgrd_weight_image = np.zeros((ny,nx),dtype=np.float32)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Loop over the image list
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    for this_fname in ProgressBar(image_list):        
        
        this_hdu_list = fits.open(this_fname)
        hdu_image = this_hdu_list['IMAGE']
        hdu_flags = this_hdu_list['FLAGS']
        hdu_zodi = this_hdu_list['ZODI']        
        image_header = hdu_image.header
        
        lam, bw = make_wavelength_image(
            hdu_list = this_hdu_list,
            use_hdu = 'IMAGE',
        )

        this_max_lam = (np.nanmax(lam + bw)).value
        this_min_lam = (np.nanmin(lam - bw)).value
        
        if (this_max_lam < min_lam):
            continue

        if (this_min_lam > max_lam):
            continue
        
        hdu_lam = fits.PrimaryHDU(lam, image_header)
        hdu_bw = fits.PrimaryHDU(bw, image_header)

        # Implement flags and subtract ZODI if requested

        mask = make_mask_from_flags(
            hdu_flags.data,
            flags_to_use = flags_to_use,
        )

        if sub_zodi:
            masked_data = hdu_image.data - hdu_zodi.data
        else:
            masked_data = hdu_image.data
            
        masked_data[mask] = np.nan

        # Fit a background (or leave it at 0.0)
        if sub_bkgrd:
            bkgrd = estimate_spherex_bkgrd(
                image = masked_data,
                header = image_header,
                lam = lam,
                bw = bw,
                gal_coord = gal_coord,
                gal_rad_deg = gal_rad_deg,
                gal_incl = gal_incl,
                gal_pa = gal_pa,
            )
        else:
            bkgrd = np.zeros_like(masked_data)

        masked_data -= bkgrd
        hdu_masked_image = fits.PrimaryHDU(masked_data, image_header)
        hdu_bkgrd_image = fits.PrimaryHDU(bkgrd, image_header)

        # This is pretty annoyingly inefficient to repeat three
        # reprojects, but for now it is what it is. Reproject image,
        # wavelength, and bandwidth to the target header.
        
        missing = np.nan
        
        reprojected_image, footprint_image = \
            reproject_interp(hdu_masked_image, target_header_2d, order='bilinear')
        reprojected_image[footprint_image == 0] = missing

        reprojected_lam, footprint_lam = \
            reproject_interp(hdu_lam, target_header_2d, order='bilinear')
        reprojected_lam[footprint_lam == 0] = missing

        reprojected_bw, footprint_bw = \
            reproject_interp(hdu_bw, target_header_2d, order='bilinear')
        reprojected_bw[footprint_bw == 0] = missing        

        reprojected_bkgrd, footprint_bkgrd = \
            reproject_interp(hdu_bkgrd_image, target_header_2d, order='bilinear')
        reprojected_bkgrd[footprint_bkgrd == 0] = missing        

        weight = np.isfinite(reprojected_image)*1.0
        
        # Identify overlap

        if continuum is not None:
            
            reprojected_image = reprojected_image - reprojected_cont

        # Identify relevant pixels

        # ... lower and upper end of the bandpass for each pixel
        low_lam = reprojected_lam - 0.5*reprojected_bw                
        hi_lam = reprojected_lam + 0.5*reprojected_bw

        # ... identify where overlap with the line starts
        # pixel-by-pixel
        
        start_im = low_lam
        start_im[np.where(min_lam > start_im)] = min_lam

        # ... identify where overlap with the line stars
        # pixel-by-pixel
        
        stop_im = hi_lam
        stop_im[np.where(max_lam < stop_im)] = max_lam

        # ... fraction of overlap with the full line
        
        overlap_im = (stop_im-start_im)/(max_lam - min_lam)

        # ... find where the filter overlaps the line above the
        # threshold
        
        y_ind, x_ind = \
            np.where(overlap_im >= frac_thresh)
        
        # Convert to line integral

        if operation.lower() == 'integrate':

            hi_freq = sol_cgs/(low_lam*1E-4)
            low_freq = sol_cgs/(hi_lam*1E-4)

            bw_freq = hi_freq-low_freq

            # Starts in MJy/sr then convert to cgs (so erg/s/cm2/Hz)
            # then multiply by bandwidth in Hz
            
            reprojected_image = \
                reprojected_image * 1E6 * \
                1E-23 * bw_freq

            target_header_2d['BUNIT'] = 'erg/s/cm**2/sr'
            
        if operation.lower() == 'average':

            # Stay in MJy/sr
            pass
        
        sum_image[y_ind, x_ind] = \
            sum_image[y_ind, x_ind] + \
            (reprojected_image[y_ind, x_ind]*weight[y_ind, x_ind])

        weight_image[y_ind, x_ind] = \
            weight_image[y_ind, x_ind] + \
            (weight[y_ind, x_ind])        

        bkgrd_sum_image[y_ind, x_ind] = \
            bkgrd_sum_image[y_ind, x_ind] + \
            (reprojected_bkgrd[y_ind, x_ind]*weight[y_ind, x_ind])
        
        bkgrd_weight_image[y_ind, x_ind] = \
            bkgrd_weight_image[y_ind, x_ind] + weight[y_ind, x_ind]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Output and return
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    image = sum_image / weight_image
    image[np.where(weight_image == 0.0)] = np.nan

    image_hdu = fits.PrimaryHDU(image, target_header_2d)
    if outfile is not None:
        image_hdu.writeto(outfile, overwrite=overwrite)
        
    weight_hdu = fits.PrimaryHDU(weight_image, target_header_2d)
    if outfile is not None:
        weight_hdu.writeto(outfile.replace('.fits','_weight.fits')
                           , overwrite=overwrite)

    bkgrd_image = bkgrd_sum_image / bkgrd_weight_image
    bkgrd_image[np.where(bkgrd_weight_image == 0.0)] = np.nan

    bkgrd_header = target_header.copy()
    bkgrd_header['BUNIT'] = 'MJy/sr'

    bkgrd_hdu = fits.PrimaryHDU(bkgrd_image, bkgrd_header)
    if outfile is not None:
        bkgrd_hdu.writeto(outfile.replace('.fits','_bkgrd.fits')
                       , overwrite=overwrite)

    bkgrd_weight_hdu = fits.PrimaryHDU(bkgrd_weight_image, bkgrd_header)
    if outfile is not None:
        bkgrd_hdu.writeto(outfile.replace('.fits','_bkgrdweight.fits')
                       , overwrite=overwrite)
    
    return(image_hdu)
    
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routine to estimate a smooth continuum
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# God this is slow - but it does work nicely. Consult on some
# speedups.

def estimate_continuum(
        cube = None,
        features_to_flag = {},
        outfile = None,
        overwrite = True):
    """
    """

    # Read the cube
    sc = SpectralCube.read(cube)
    vaxis = sc.spectral_axis
    n_z, n_y, n_x = sc.shape

    # Flag the lines in the cube (fix the broadcasting here later)
    cube = sc.filled_data[:]
    for this_lam, this_bw in features_to_flag.items():
        mask_1d = np.abs(this_lam - vaxis.value) <= 0.5 * this_bw
        mask_3d = np.broadcast_to(mask_1d[:,np.newaxis,np.newaxis],sc.shape)
        cube[mask_3d] = np.nan
    
    # Loop over the spectra
    cont_cube = np.zeros((n_z, n_y, n_x), dtype=np.float32)*np.nan
    for yy in ProgressBar(range(n_y)):
        for xx in range(n_x):
            this_spec = (cube[:,yy,xx]).flatten()
            ind = np.where(np.isfinite(this_spec))
            this_x = vaxis[ind]
            this_y = this_spec[ind]
            cs = make_smoothing_spline(this_x, this_y, lam=0.01)
            pred_y = cs(vaxis)
            max_chan = np.max(ind)
            cont_cube[:max_chan,yy,xx] = pred_y[:max_chan]

    # Write the output
    cont_hdu = fits.PrimaryHDU(cont_cube, sc.header)
    if outfile is not None:
        cont_hdu.writeto(outfile, overwrite=True)
    return(cont_hdu)

            
