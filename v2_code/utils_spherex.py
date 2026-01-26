#!/usr/bin/env python3

import os, glob
import numpy as np
import warnings

import urllib.request
import urllib.error

from tqdm import tqdm

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.table import Table, QTable
from astropy import units as u, constants as const

# Astroquery for IRSA access
from astroquery.ipac.irsa import Irsa

# Reproject for image alignment
from reproject import reproject_interp

# Used to analyze the cube
from numpy import linalg
from scipy.interpolate import make_smoothing_spline
from spectral_cube import SpectralCube

#import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
#import astropy.utils.exceptions
#warnings.simplefilter('ignore', category=astropy.utils.exceptions.AstropyWarning)    


# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Related to deprojection
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


def deproject(
        center_coord=None,
        incl=0*u.deg,
        pa=0*u.deg,
        template_header=None,
        template_wcs=None,
        template_naxis=None,
        template_ra=None,
        template_dec=None,
        return_offset=False,
        verbose=False):
    """Calculate deprojected radii and projected angles in a disk.

    This function deals with projected images of astronomical objects
    with an intrinsic disk geometry. Given sky coordinates of the disk
    center, disk inclination and position angle, this function
    calculates deprojected radii and projected angles based on

    (1) a FITS header (`header`), or

    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    
    (3) RA and DEC coodinates (`ra` + `dec`).
    
    Both deprojected radii and projected angles are defined relative
    to the center in the inclined disk frame. For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.

    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` object or array-like
        Sky coordinates of the disk center
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    header : `~astropy.io.fits.Header` object, optional
        FITS header specifying the WCS and size of the output 2D maps
    wcs : `~astropy.wcs.WCS` object, optional
        WCS of the output 2D maps
    naxis : array-like (with two elements), optional
        Size of the output 2D maps
    ra : array-like, optional
        RA coordinate of the sky locations of interest
    dec : array-like, optional
        DEC coordinate of the sky locations of interest
    return_offset : bool, optional
        Whether to return the angular offset coordinates together with
        deprojected radii and angles. Default is to not return.

    Returns
    -------
    deprojected coordinates : list of arrays
        If `return_offset` is set to True, the returned arrays include
        deprojected radii, projected angles, as well as angular offset
        coordinates along East-West and North-South direction;
        otherwise only the former two arrays will be returned.

    Notes
    -----
    This is the Python version of an IDL function `deproject` included in the `cpropstoo` package. See URL below:

    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro

    Convention on the in-plane position angle w.r.t. the receding node may be flipped.

    Python routine from Jiayi Sun. Modified for compatibility with
    z0mgs namespace.

    """

    if isinstance(center_coord, SkyCoord):
        x0_deg = center_coord.ra.degree
        y0_deg = center_coord.dec.degree
    else:
        x0_deg, y0_deg = center_coord
        if hasattr(x0_deg, 'unit'):
            x0_deg = x0_deg.to(u.deg).value
            y0_deg = y0_deg.to(u.deg).value
    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    if template_header is not None:
        wcs_cel = WCS(template_header).celestial
        naxis1 = template_header['NAXIS1']
        naxis2 = template_header['NAXIS2']
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    elif (template_wcs is not None) and \
         (template_naxis is not None):
        wcs_cel = template_wcs.celestial
        naxis1, naxis2 = template_naxis
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    else:
        if template_ra.ndim == 1:
            ra_deg, dec_deg = \
                np.broadcast_arrays(template_ra, template_dec)
        else:
            ra_deg, dec_deg = template_ra, template_dec
            if verbose:
                print("ra ndim != 1")
        if hasattr(template_ra, 'unit'):
            ra_deg = template_ra.to(u.deg).value
            dec_deg = template_dec.to(u.deg).value
    
    
    #else:
        #ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        #if hasattr(ra_deg, 'unit'):
            #ra_deg = ra_deg.to(u.deg).value
            #dec_deg = dec_deg.to(u.deg).value

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dy_deg = dec_deg - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) +
                    dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) -
                    dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_offset:
        return radius_deg, projang_deg, deprojdx_deg , deprojdy_deg
    else:
        return radius_deg, projang_deg
    

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Manage targets
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


def write_template_tab(
        outfile='example_spherex_targs.ecsv',
        ):

    coords = SkyCoord.from_name('ngc5194')
    
    targ = {
        'gal':'ngc5194',
        'ra':coords.ra,
        'dec':coords.dec,
        'fov':20.*u.arcmin,
        'mask_rad':10.*u.arcmin,
        'mask_pa':0.0*u.deg,
        'mask_incl':0.0*u.deg,
        'vrad':463*u.km/u.s,
        'vwidth':500.*u.km/u.s,
    }

    tab = Table([targ])
    tab.write(outfile, overwrite=True, delimiter=',',
              format='ascii.ecsv')

    return(tab)

def write_sample_tab(
        outfile=None,
        galbase='/home/leroy.42/idl/galbase/gal_data/gal_base.fits',
        tags=[],
        just_gals=[],
):
    """Helper function to write tables for SPHEREx pipeline. Requires a
    specific database, so may not be generally useful.
    """

    gb = Table.read(galbase)

    # Identify the targets
    
    if (len(tags) == 0) & (len(just_gals) == 0):
        return(None)

    gb['KEEP_ROW'] = False
    for this_row in ProgressBar(gb):

        if len(tags) > 0:
            for this_tag in tags:
                if this_tag in this_row['TAGS']:
                    this_row['KEEP_ROW'] = True

        if len(just_gals) > 0:
            if this_row['OBJNAME'].strip().lower() in just_gals:
                this_row['KEEP_ROW'] = True

    gb = gb[gb['KEEP_ROW'] == True]

    # Make the table
    
    targs = []

    for this_row in gb:
        targ = {
            'gal':str(this_row['OBJNAME']).strip().lower(),
            'ra':float(this_row['RA_DEG'])*u.deg,
            'dec':float(this_row['DEC_DEG'])*u.deg,
            'fov':float(this_row['R25_DEG'])*2*60.*u.arcmin,
            'mask_rad':float(this_row['R25_DEG'])*1.5*60.*u.arcmin,
            'mask_pa':0.0*u.deg,
            'mask_incl':0.0*u.deg,
            'vrad':float(this_row['VHEL_KMS'])*u.km/u.s,
            'vwidth':float(this_row['VMAXG_KMS']*2.0)*u.km/u.s,
        }

        if np.isfinite(targ['vwidth']) == False:
            targ['vwidth'] = 400.*u.km/u.s

        if np.isfinite(targ['vwidth']) == False:
            targ['vhel_kms'] = 0.0*u.km/u.s
        
        targs.append(targ)

    targ_tab = Table(targs)

    
    
    if outfile is not None:
        targ_tab.write(outfile, overwrite=True, delimiter=',',
                       format='ascii.ecsv')
    
    print(targ_tab)
        
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
    
    for this_fname in tqdm(image_list):

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


def make_mask_from_flags(
        flag_image,
        flags_to_use = [
            'SUR_ERROR', 'NONFUNC', 'MISSING_DATA',
            'HOT', 'COLD', 'NONLINEAR', 'PERSIST'],
        ):
    """Given a list of flag names to use construct a mask from the input
    flag image. The flag image is a bitmask, and the mapping between
    flags and bits is defined in the explanatory supplement and coded
    into a dictionary above.

    Parameters
    ----------
    flag_image : array
        Input flag image where each pixel value is a bitmask
    flags_to_use : list of str
        List of flag names to include in the mask
        
    Returns
    -------
    mask : bool array
        Boolean mask where True indicates flagged pixels

    Notes
    -----
    Suggested flags for background estimation masking:
        OVERFLOW, SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, 
        NONLINEAR, PERSIST, OUTLIER, SOURCE, TRANSIENT

    Suggested flags for on-source photometry:
        SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR, PERSIST
    """

    # Get bit positions for requested flags
    flag_dict = spherex_flag_dict()

    # Combine requested flags into a single bitmask
    combined_bitmask = 0
    for this_flag in flags_to_use:
        try:
            bit_position = flag_dict[this_flag.upper()]
            # OR together the bit values (2^bit_position)
            combined_bitmask |= (1 << bit_position)
        except KeyError:
            print(f"Warning: Flag '{this_flag}' not recognized -- ignore")
            continue
    
    # Single bitwise AND to check all requested flags at once
    mask = (flag_image & combined_bitmask) != 0
    
    return mask


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
    
    for this_fname in tqdm(image_list):        
        
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
        ext_to_use = 'BKSUB',        
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
    for this_fname in tqdm(image_list):   

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
        lam_min = 0.7, lam_max = 5.2, lam_step = 0.0075,
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

        finite_lam = np.isfinite(lam_cube)
        finite_val = np.isfinite(int_cube)
        
        for zz in tqdm(range(nz)):

            this_lam = lam_array[zz]
            
            # Compare the wavelength at each pixel to the center of
            # the current channel in units of bandwidth
            
            delta = np.abs(this_lam - lam_cube)/bw_cube
            
            # Keep the pixels where that difference is less than half
            # the bandwidth
            
            sed_ind, y_ind, x_ind = \
                np.where((delta <= 0.5)*finite_lam*finite_val)

            val_vec = int_cube[sed_ind, y_ind, x_ind]
            bw_vec = bw_cube[sed_ind, y_ind, x_ind]
            wt_vec = val_vec*0.0 + 1.0
            
            z_ind = np.zeros_like(y_ind,dtype=int)+zz

            np.add.at(sum_cube, (z_ind, y_ind, x_ind), val_vec*wt_vec)
            np.add.at(bw_sum_cube, (z_ind, y_ind, x_ind), bw_vec*wt_vec)
            np.add.at(weight_cube, (z_ind, y_ind, x_ind), wt_vec)            

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
# Routine to estimate a smooth continuum
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


def estimate_continuum(
        # Input cubes
        int_cube = None,
        lam_cube = None,
        bw_cube = None,
        # Desired wavelength grid
        lam_min = 0.7, lam_max = 5.2, lam_step = 0.0075,
        lam_unit = 'um',
        # Features to flag out
        features_to_flag = [],
        feature_dict = {},
        # Redshift
        vrad = 0.0*u.km/u.s,
        vwidth = 0.0*u.km/u.s,
        # Output
        outfile_cube = None,
        outfile_seds = None,
        overwrite=True):
    """
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

    nsed, nysed, nxsed = int_cube.shape

    # Doppler shift for source redshift and width

    sol_kms = 2.99792E5*u.km/u.s
    sol_cgs = 2.99792468E10*u.cm/u.s
    
    dopp_fac = \
        np.sqrt((1.0+vrad/sol_kms)/(1.0-vrad/sol_kms))
    dopp_fac_high = \
        np.sqrt((1.0+(vrad+vwidth)/sol_kms)/(1.0-(vrad+vwidth)/sol_kms))
    dopp_fac_low = \
        np.sqrt((1.0+(vrad-vwidth)/sol_kms)/(1.0-(vrad-vwidth)/sol_kms))
    dopp_fac_width = dopp_fac_high - dopp_fac_low
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Flag features
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    for this_feature in features_to_flag:

        if this_feature not in feature_dict:
            continue

        # Identify where the wavelength of a data point is within half
        # the bandwidth plus half the feature width of the redshifted
        # line frequency
        
        this_feature_lam = feature_dict[this_feature]['lam']
        this_feature_width = feature_dict[this_feature]['width']
        
        delta = np.abs((this_feature_lam * dopp_fac).to(u.um).value - lam_cube)
        width = (dopp_fac_width * this_feature_lam).to(u.um).value + \
            bw_cube * 0.5 + \
            this_feature_width.to(u.um).value * 0.5

        line_ind = np.where(delta <= width)

        # Blank the intensity here but leave the line width
        int_cube[line_ind] = np.nan
        
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Calculate a smooth SED
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Initialize the header and output

    # ... one output is on the SED grid
    sed_header = int_hdu.header.copy()

    # ... the other on a regular grid
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

    # Initialize an output cube
    cont_sed = np.zeros((nsed, ny, nx), dtype=np.float32)*np.nan
    cont_cube = np.zeros((nz, ny, nx), dtype=np.float32)*np.nan

    # Loop over the spectra
    for yy in tqdm(range(ny)):
        for xx in range(nx):
            
            # Get this SED
            this_lam = (lam_cube[:,yy,xx]).flatten()
            this_spec = (int_cube[:,yy,xx]).flatten()
            
            ind = np.where(np.isfinite(this_spec)*np.isfinite(this_lam))
            this_x = this_lam[ind]
            this_y = this_spec[ind]

            sort_ind = np.argsort(this_x)
            this_x = this_x[sort_ind]
            this_y = this_y[sort_ind]

            # Annoying error catch - the splines want an ascending x
            # array and sometimes you end up with identical
            # wavelengths
            
            delta_x = this_x - np.roll(this_x,1)
            delta_x[0] = 1.0
            keep_ind = np.where(delta_x > 0.0)
            this_x = this_x[keep_ind]
            this_y = this_y[keep_ind]

            # Fit a spline
            
            cs = make_smoothing_spline(this_x, this_y, lam=0.02)

            # Predict the values in the cube

            pred_y = cs(lam_array)

            # ... avoid extrapolation
            bad_ind = np.where((lam_array < np.nanmin(this_x)) |
                               (lam_array > np.nanmax(this_x)))
            pred_y[bad_ind] = np.nan

            # ... fill in the output
            cont_cube[:,yy,xx] = pred_y

            # Predict SED values (this_lam is the SED wavelengths)

            pred_sed = cs(this_lam)

            # ... avoid extrapolation
            bad_ind = np.where((this_lam < np.nanmin(this_x)) |
                               (this_lam > np.nanmax(this_x)))
            pred_sed[bad_ind] = np.nan

            # ... fill in the output
            cont_sed[:,yy,xx] = pred_sed

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Write the output
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    cont_cube_hdu = fits.PrimaryHDU(cont_cube, target_header)    
    cont_sed_hdu = fits.PrimaryHDU(cont_sed, sed_header)
        
    if outfile_cube is not None:
        cont_cube_hdu.writeto(outfile_cube, overwrite=True)

    if outfile_seds is not None:
        cont_sed_hdu.writeto(outfile_seds, overwrite=True)
        
    return((cont_cube_hdu, cont_sed_hdu))


def median_filter_irregular(x, y, x_width, show_progress=True):
    """
    Apply median filter to irregularly sampled data using x-space window.
    
    For each point at position x[i], computes the median of all y values
    within the x-range [x[i] - x_width/2, x[i] + x_width/2].
    
    NaN values are automatically excluded from the median calculation.
    
    Parameters:
    -----------
    x : (nlam, npix) array
        Sample positions
    y : (nlam, npix) array
        Sample values
    x_width : float
        Width of the x-window for median filtering
    show_progress : bool
        Whether to show a progress bar during pixel-wise filtering.
    
    Returns:
    --------
    y_filtered : (nlam, npix) array
        Median-filtered values
    """
    nlam, npix = x.shape
    y_filtered = np.empty_like(y)
    half_width = x_width / 2
    # Loop over each pixel
    for ipix in tqdm(range(npix), disable=(not show_progress)):
        # Compute pairwise distances: (nlam, nlam)
        dist_matrix = np.abs(x[:, ipix][:, None] - x[:, ipix][None, :])
        # Create distance-based mask: (nlam, nlam)
        distance_mask = dist_matrix > half_width
        # Create NaN mask: (nlam, nlam) - True where y values are NaN or x values are NaN
        nan_mask = (np.isnan(y[:, ipix]) | np.isnan(x[:, ipix]))[None, :]
        # Combine masks: mask out points that are too far OR have NaN values
        combined_mask = distance_mask | nan_mask
        # For each point i, compute median of y[j] where mask[i,j] is False
        # Use masked array to handle variable number of neighbors
        y_masked = np.ma.array(np.tile(y[:, ipix], (nlam, 1)), mask=combined_mask)
        y_filtered[:, ipix] = np.ma.median(y_masked, axis=1).filled(np.nan)
    return y_filtered


def decide_fourier_grid(
        xlim, fwhm, oversample=2, decay=1e-2, bandwidth_fraction=1.0):
    """
    Estimate max frequency and number of components from domain size and resolution.
    - bandwidth_fraction: < 1 for low-pass filtering
    """
    sigma = float(np.nanmedian(fwhm) / 2.3548)  # average sigma from FWHM
    omega_max = np.sqrt(2.0 * np.log(1.0 / decay)) / sigma
    omega_max *= bandwidth_fraction             # scale bandwidth
    N_comp = int(np.ceil((omega_max * (xlim[1] - xlim[0])) / np.pi))
    N_comp = max(8, int(oversample * N_comp))   # floor to avoid tiny N
    return float(omega_max), int(N_comp)


def reconstruct_fls(
    x, y, x_bw, weight,
    # omega grid parameters
    xlim=None,
    bandwidth_fraction=1.0,
    # ridge regularization
    lam0=1e-2,
    # progress bar
    verbose=True):
    """
    Spectral reconstruction from irregular samples using Fourier Least Squares.

    Accounts for per-sample Gaussian LSFs.
    Uses basic ridge regularization.
    
    Parameters:
    -----------
    x : (nlam, npix) array
        Sample positions
    y : (nlam, npix) array
        Sample values
    x_bw : (nlam, npix) array
        Per-sample FWHM (in x-units)
    weight : (nlam, npix) array, optional
        Per-sample weights (e.g., 1/variance). If None, estimated from noise.
    xlim : tuple of (xmin, xmax), optional
        Range for determining Fourier grid. If None, uses min/max of x.
    bandwidth_fraction : float
        Fraction of full bandwidth to use (< 1.0 for low-pass filtering)
    lam0 : float
        Ridge regularization strength
    show_progress : bool
        Whether to show a progress bar during pixel-wise solving.

    Returns:
    --------
    model : callable
        Function that takes x_out (mlam,) array and returns y_out (mlam, npix) array
    """
    nlam, npix = x.shape

    # Compute Fourier space grid
    if verbose:
        print("  Deciding Fourier grid...")
    if xlim is None:
        xlim = np.nanmin(x), np.nanmax(x)
    omega_max, ncomp = decide_fourier_grid(
        xlim, np.nanmean(x_bw),
        bandwidth_fraction=bandwidth_fraction)
    omega = np.linspace(-omega_max, omega_max, ncomp)

    # Forward model
    if verbose:
        print("  Building forward model...")
    sigma = x_bw / 2.3548  # (nlam,npix)
    H = np.exp(-0.5 * (sigma[..., None] * omega[None, None, :])**2)  # (nlam,npix,ncomp)
    Phi = np.exp(1j * x[..., None] * omega[None, None, :])           # (nlam,npix,ncomp)
    A = H * Phi  # (nlam,npix,ncomp)

    # Apply weights
    if verbose:
        print("  Applying weights...")
    A = A * weight[..., None]      # (nlam,npix,ncomp)
    y_eff = y * weight       # (nlam,npix)

    # Normal equations per batch
    if verbose:
        print("  Normalizing equations...")
    ATA = np.einsum('nbk,nbl->bkl', A.conj(), A)   # (npix,ncomp,ncomp)
    ATy = np.einsum('nbk,nb->bk',  A.conj(), y_eff) # (npix,ncomp)

    # Ridge regularization
    ATA += lam0 * np.eye(ncomp)[None, :, :]

    # Solve per batch
    if verbose:
        print("  Solving for Fourier coefficients...")
    solutions = np.empty((npix, ncomp), dtype=complex)
    for ipix in range(npix):
        solutions[ipix] = linalg.solve(ATA[ipix], ATy[ipix])

    # Return a function that evaluates on any output grid
    def model(x_out):
        """
        Evaluate the reconstructed spectrum at given positions.
        
        Parameters:
        -----------
        x_out : (mlam,) or (mlam, npix) array
            Output grid positions. If 1D, same grid is used for all pixels.
            If 2D, each pixel can have a different output grid.
            
        Returns:
        --------
        y_out : (mlam, npix) array
            Reconstructed signal on target grid x_out
        """
        x_out = np.asarray(x_out)
        if x_out.ndim == 1:
            # Same grid for all pixels
            Phi_g = np.exp(1j * x_out[:, None] * omega[None, :])  # (mlam,ncomp)
            y_out = np.real(Phi_g @ solutions.T)  # (mlam,npix)
            # Avoid extrapolation: mask values outside training range
            x_min, x_max = np.nanmin(x), np.nanmax(x)
            extrapolation_mask = (x_out < x_min) | (x_out > x_max)
            y_out[extrapolation_mask, :] = np.nan
        elif x_out.ndim == 2:
            # Different grid for each pixel
            mlam, npix_out = x_out.shape
            if npix_out != npix:
                raise ValueError(f"x_out has {npix_out} pixels but model has {npix} pixels")
            Phi_g = np.exp(1j * x_out[:, :, None] * omega[None, None, :])  # (mlam,npix,ncomp)
            y_out = np.real(np.einsum('mpk,pk->mp', Phi_g, solutions))  # (mlam,npix)
            # Avoid extrapolation: mask values outside training range per pixel
            x_min = np.nanmin(x, axis=0)  # (npix,)
            x_max = np.nanmax(x, axis=0)  # (npix,)
            extrapolation_mask = (x_out < x_min[None, :]) | (x_out > x_max[None, :])  # (mlam,npix)
            y_out[extrapolation_mask] = np.nan
        else:
            raise ValueError(f"x_out must be 1D or 2D, got shape {x_out.shape}")
        return y_out
    
    return model


def estimate_continuum_fls(
        # Input cubes
        int_cube = None,
        lam_cube = None,
        bw_cube = None,
        # Features to flag out
        features_to_flag = [],
        feature_dict = {},
        # Redshift
        vrad = 0.0*u.km/u.s,
        vwidth = 0.0*u.km/u.s,
        # Width of median filter
        filter_width = None,
        # Desired wavelength grid
        lam_min = 0.7,
        lam_max = 5.2,
        lam_step = 0.0075,
        lam_unit = 'um',
        # Output
        outfile_seds = None,
        outfile_cube = None,
        overwrite=True,
        # arguments for FLS
        bandwidth_fraction=0.15,
        verbose=True,
        **kwargs):
    """
    Estimate continuum from a SED cube using Fourier Least Squares method.
    
    Uses a Fourier Least Squares approach to fit a smooth continuum to
    irregularly sampled spectral data while accounting for per-sample LSFs.
    Line features can be masked before fitting. The continuum is evaluated
    on both the original SED grid and/or a regular wavelength grid.
    
    Parameters
    ----------
    int_cube : str
        Path to FITS file containing intensity cube (nlam, ny, nx)
    lam_cube : str, optional
        Path to wavelength cube. If None, derived from int_cube filename
    bw_cube : str, optional
        Path to bandwidth cube. If None, derived from int_cube filename
    features_to_flag : list of str
        Names of spectral features to mask before continuum fitting
    feature_dict : dict
        Dictionary mapping feature names to dicts with 'lam' and 'width' keys
    vrad : Quantity
        Radial velocity of the target
    vwidth : Quantity
        Velocity width of the target
    filter_width : Quantity or None
        Width of median filter window (default None for no filtering)
    lam_min : float
        Minimum wavelength for regular output grid
    lam_max : float
        Maximum wavelength for regular output grid
    lam_step : float
        Wavelength step for regular output grid
    lam_unit : str
        Unit for wavelength axis in output (default 'um')
    outfile_seds : str, optional
        Output file for continuum evaluated on original SED grid
    outfile_cube : str, optional
        Output file for continuum on regular wavelength grid
    overwrite : bool
        Whether to overwrite existing output files (default True)
    bandwidth_fraction : float
        Fraction of full Fourier bandwidth to use. Values < 1 act as
        low-pass filter for smoother continuum (default 0.15)
    verbose : bool
        Whether to print progress messages
    **kwargs : dict
        Additional arguments passed to reconstruct_fls()
    
    Returns
    -------
    fls_model : callable
        Function that takes wavelength array and returns continuum values.
        Can be called with 1D array (mlam,) or 2D array (mlam, npix).
    
    Notes
    -----
    The FLS method accounts for per-pixel variations in wavelength sampling
    and bandwidth. It is particularly effective for handling irregular
    sampling and provides smooth interpolation without strong assumptions
    about the functional form of the continuum.
    """
    
    if lam_cube is None:
        lam_cube = str(int_cube).replace('.fits','_lam.fits')        
        
    if bw_cube is None:
        bw_cube = str(int_cube).replace('.fits','_bw.fits')

    lam_data, hdr = fits.getdata(lam_cube, header=True)
    assert hdr['BUNIT'] == 'um'
    bw_data, hdr = fits.getdata(bw_cube, header=True)
    assert hdr['BUNIT'] == 'um'
    int_data, hdr = fits.getdata(int_cube, header=True)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Flag features
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if verbose:
        print("  Flagging line features...")

    # Doppler shift for source redshift and width
    dopp_fac = np.sqrt((1.0+vrad/const.c)/(1.0-vrad/const.c))
    dopp_fac_high = np.sqrt(
        (1.0+(vrad+vwidth)/const.c)/(1.0-(vrad+vwidth)/const.c))
    dopp_fac_low = np.sqrt(
        (1.0+(vrad-vwidth)/const.c)/(1.0-(vrad-vwidth)/const.c))
    dopp_fac_width = dopp_fac_high - dopp_fac_low

    for this_feature in features_to_flag:

        if verbose and (this_feature not in feature_dict):
            print(f"Warning: Feature '{this_feature}' not recognized -- ignore")
            continue
        
        this_feature_lam = feature_dict[this_feature]['lam']
        this_feature_width = feature_dict[this_feature]['width']

        # Identify where the wavelength of a data point overlaps with
        # the line (considering redshift, bandwidth, and line width)
        abs_delta_lam = np.abs(
            (this_feature_lam * dopp_fac).to(u.um).value - lam_data)
        window_width = np.sqrt(
            (dopp_fac_width * this_feature_lam).to(u.um).value**2 +
            bw_data**2 +
            this_feature_width.to(u.um).value**2)

        # Blank the intensity here
        int_data[abs_delta_lam <= window_width] = np.nan

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Prepare data for FLS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if verbose:
        print("  Organizing data for FLS...")

    # reshape to 2D arrays (wavelength, pixel)
    nlam, ny, nx = int_data.shape
    int_data = int_data.reshape((nlam, ny*nx))
    lam_data = lam_data.reshape((nlam, ny*nx))
    bw_data = bw_data.reshape((nlam, ny*nx))

    # sort by wavelength
    sort_idx = np.argsort(lam_data, axis=0)
    lam_data = np.take_along_axis(lam_data, sort_idx, axis=0)
    int_data = np.take_along_axis(int_data, sort_idx, axis=0)
    bw_data = np.take_along_axis(bw_data, sort_idx, axis=0)

    # estimate noise per spectrum
    noise_std = np.nanmedian(np.abs(np.diff(int_data, axis=0)), axis=0)
    noise_std /= 0.6745 * np.sqrt(2)

    # apply median filter to remove outliers
    if filter_width is not None:
        if verbose:
            print("  Applying median filter...")
        int_data = median_filter_irregular(
            lam_data, int_data,
            x_width=filter_width.to('um').value,
            show_progress=verbose)
    
    # assign weights and handle NaNs
    weights = np.isfinite(int_data).astype(np.float32)
    weights /= noise_std[None, :]**2
    int_data[np.isnan(int_data)] = 0.0
    lam_data[np.isnan(lam_data)] = 0.0
    bw_data[np.isnan(bw_data)] = 1.0

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Model continuum with FLS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    fls_model = reconstruct_fls(
        lam_data, int_data, bw_data, weights,
        bandwidth_fraction=bandwidth_fraction, **kwargs)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Evaluate on output grids
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if outfile_seds is not None:
        if verbose:
            print("  Evaluating on SED grid...")
        # evaluate on SED grid
        cont_sed = fls_model(lam_data)
        cont_sed[lam_data == 0.0] = np.nan
        # restore original order
        inv_sort_idx = np.argsort(sort_idx, axis=0)
        cont_sed = np.take_along_axis(
            cont_sed, inv_sort_idx, axis=0)
        # reshape to cube
        cont_sed = cont_sed.reshape((nlam, ny, nx))
        # write to file
        cont_sed_hdu = fits.PrimaryHDU(cont_sed.astype(np.float32), hdr)
        cont_sed_hdu.writeto(outfile_seds, overwrite=overwrite)

    if outfile_cube is not None:
        if verbose:
            print("  Evaluating on cube grid...")
        # create regular output grid
        lam_out = np.arange(lam_min, lam_max + lam_step, lam_step)
        hdr['NAXIS'] = 3
        hdr['NAXIS3'] = len(lam_out)
        hdr['CTYPE3'] = 'WAVE'
        hdr['CUNIT3'] = lam_unit
        hdr['CRPIX3'] = 1
        hdr['CRVAL3'] = lam_out[0]
        hdr['CDELT3'] = lam_step
        # evaluate model
        cont_cube = fls_model(lam_out).reshape((len(lam_out), ny, nx))
        # write to file
        cont_cube_hdu = fits.PrimaryHDU(cont_cube.astype(np.float32), hdr)
        cont_cube_hdu.writeto(outfile_cube, overwrite=overwrite)

    return fls_model

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routine to integrate the continuum
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def integrate_continuum(
        # Input cube
        cont_cube = None,
        # Wavelength range
        lam_min = None,
        lam_max = None,
        # Output file
        outfile= None,
        overwrite= True,
        ):
    """
    """
    
    cube = SpectralCube.read(cont_cube)

    lam_axis = cube.spectral_axis
    if lam_min is None:
        np.nanmin(lam_axis)
    if lam_max is None:
        np.nanmax(lam_axis)
        
    subcube = cube.spectral_slab(lam_min, lam_max)

    hdr_2d = cube.header.copy()
    hdr_2d['NAXIS'] = 2
    del hdr_2d['NAXIS3']
    del hdr_2d['CRVAL3']
    del hdr_2d['CDELT3']
    del hdr_2d['CRPIX3']
    del hdr_2d['CTYPE3']
    del hdr_2d['CUNIT3']

    image = subcube.median(axis=0).filled_data[:].value
    print(type(image))

    hdu = fits.PrimaryHDU(data = image, header = hdr_2d)
    if outfile is not None:
        hdu.writeto(outfile, overwrite=overwrite)

    return(hdu)
    
    
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routine to make a line map
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


def make_spherex_line_image(
        # Input cubes
        int_cube = None,
        cont_cube = None,
        lam_cube = None,
        bw_cube = None,
        # Feature
        feature_dict = {},
        # Source redshift
        vrad = 0.0*u.km/u.s,
        #vwidth = 0.0*u.km/u.s,  # Not used yet
        # Assumed LSF shape
        lsf = 'tophat',
        # Output
        outfile = None,
        overwrite = True):
    """
    Create a line-integrated image.
    """

    if lam_cube is None:
        lam_cube = str(int_cube).replace('.fits', '_lam.fits')        
        
    if bw_cube is None:
        bw_cube = str(int_cube).replace('.fits', '_bw.fits')

    # calculate line center wavelength (assuming 0 width for now)
    line_lam_rest = feature_dict['lam']
    line_lam_obs = line_lam_rest * (1 + (vrad / const.c).to('').value)

    # zoom in on relevant data by sorting and selecting nearest channels
    # First, load only wavelength cube to determine sorting order
    with fits.open(lam_cube) as hdul:

        hdr = hdul[0].header.copy()
        u_lam = u.Unit(hdul[0].header['BUNIT'])
        lam_full = hdul[0].data * u_lam
        
        # Calculate wavelength distance from line for all pixels
        abs_delta_lam_full = np.abs(lam_full - line_lam_obs)
        
        # Set number of nearest channels to keep (within +-0.3um of the line)
        n_channels = (abs_delta_lam_full < 0.3*u.um).sum(axis=0).max()
        n_channels = np.max([10, n_channels])
        
        # Sort and index along spectral axis
        sort_idx = np.argsort(
            abs_delta_lam_full.value, axis=0)[:n_channels, :, :]
        abs_delta_lam = np.take_along_axis(
            abs_delta_lam_full.value, sort_idx, axis=0) * u_lam
        
        # Clean up large arrays
        del lam_full, abs_delta_lam_full
    
    # Now load and subselect each cube separately to limit memory use
    with fits.open(int_cube) as hdul:
        u_int = u.Unit(hdul[0].header['BUNIT'])
        int_data = np.take_along_axis(
            hdul[0].data, sort_idx, axis=0) * u_int
    
    with fits.open(bw_cube) as hdul:
        u_bw = u.Unit(hdul[0].header['BUNIT'])
        bw_data = np.take_along_axis(
            hdul[0].data, sort_idx, axis=0) * u_bw
    
    with fits.open(cont_cube) as hdul:
        cont_data = np.take_along_axis(
            hdul[0].data, sort_idx, axis=0) * u_int
    
    # subtract continuum
    line_data = int_data - cont_data

    # find bandwidth in the closest channel to the line along each sightline
    bw_line_lam = bw_data[0, :, :]

    # integrate according to LSF
    if lsf == 'tophat':

        # find all data points where line falls within the tophat LSF
        lsf_mask = (
            (abs_delta_lam <= (0.5 * bw_line_lam)) & np.isfinite(line_data))
        
        # simple average over relevant data points
        line_image = (
            np.nansum(line_data * lsf_mask, axis=0) /
            np.nansum(lsf_mask, axis=0))
        
        # convert unit
        bw_line_nu = (
            bw_line_lam * const.c / line_lam_obs**2).to('Hz')
        line_image *= bw_line_nu
        
    elif lsf == 'gaussian':

        # calculate sigma in wavelength units
        sigma_lam = bw_line_lam / np.sqrt(8 * np.log(2))

        # find all data points within 2 sigma of the line center
        lsf_mask = (
            (abs_delta_lam <= (2.0 * sigma_lam)) & np.isfinite(line_data))

        # calculate Gaussian LSF response for all data points
        lsf_curve = (
            np.exp(-0.5 * (abs_delta_lam / sigma_lam)**2) /
            np.sqrt(2 * np.pi * sigma_lam**2)).to('um-1')

        # weighted average over relevant data points
        # (assuming uniform noise on all data points, and assign weights
        # according to the noise level after scaling by the LSF)
        line_image = (
            np.nansum(line_data * lsf_curve * lsf_mask, axis=0) /
            np.nansum(lsf_curve**2 * lsf_mask, axis=0))

        # convert unit
        line_image *= const.c / line_lam_obs**2
        
    else:

        raise ValueError(
            f"LSF {lsf} not recognized. Use 'tophat' or 'gaussian'.")
    
    # prepare header
    hdr_2d = hdr.copy()
    hdr_2d['NAXIS'] = 2
    del hdr_2d['NAXIS3']
    del hdr_2d['CRVAL3']
    del hdr_2d['CDELT3']
    del hdr_2d['CRPIX3']
    del hdr_2d['CTYPE3']
    del hdr_2d['CUNIT3']
    hdr_2d['BUNIT'] = 'erg s-1 cm-2 sr-1'
    line_image = line_image.to('erg s-1 cm-2 sr-1').value
    image_hdu = fits.PrimaryHDU(line_image, hdr_2d)

    # write to file
    if outfile is not None:
        image_hdu.writeto(outfile, overwrite=overwrite)

    return image_hdu


def make_spherex_pah_image_naive(
        # Input cubes
        int_cube = None,
        cont_cube = None,
        # PAH wavelength range
        lam_min = 3.1*u.um,
        lam_max = 3.6*u.um,
        # NaN handling
        nan_policy = 'interp',  # 'interp' or 'ignore'
        # Output
        outfile = None,
        overwrite = True):
    """
    Create an integrated PAH emission image from SPHEREx spectral cubes.

    Note that this only works on the regularly gridded SPHEREx cubes!
    
    This function extracts a spectral slab over a specified wavelength range
    (default 3.1-3.6 um to capture the 3.3 um PAH feature), subtracts the
    continuum, and integrates along the spectral axis to produce a 2D image
    of PAH surface brightness. NaN values can be handled either by linear
    interpolation or by replacing them with zeros.
    
    Parameters
    ----------
    int_cube : str
        Path to the gridded spectral cube
    cont_cube : str
        Path to the gridded continuum spectral cube
    lam_min : Quantity, optional
        Minimum wavelength for spectral extraction (default: 3.1 um)
    lam_max : Quantity, optional
        Maximum wavelength for spectral extraction (default: 3.6 um)
    nan_policy : str, optional
        How to handle NaN values in the data:
        - 'interp': Linearly interpolate over NaNs along each spectrum
        - 'ignore': Replace NaNs with zeros
        Default is 'interp'
    outfile : str, optional
        Output FITS file path. If None, file is not written to disk
    overwrite : bool, optional
        Whether to overwrite existing output file (default: True)
    
    Returns
    -------
    pah_image : Projection
        2D PAH emission image in units of erg s^-1 cm^-2 sr^-1
    """

    # read in SPHEREx cubes
    cube = SpectralCube.read(int_cube)
    subcube = cube.spectral_slab(lam_min, lam_max)

    cube_cont = SpectralCube.read(cont_cube)
    subcube_cont = cube_cont.spectral_slab(lam_min, lam_max)

    # continuum subtraction
    pah_cube = subcube - subcube_cont

    if nan_policy == 'interp':
        # interpolate over NaNs
        pah_data = pah_cube.unmasked_data[:].value
        spectral_axis = pah_cube.spectral_axis
        for i in tqdm(range(pah_data.shape[1]), desc=f'Interpolating...'):
            for j in range(pah_data.shape[2]):
                spectrum = pah_data[:, i, j]
                valid = np.isfinite(spectrum)
                if valid.sum() > 1 and not valid.all():
                    # Interpolate NaN values
                    spectrum[~valid] = np.interp(
                        spectral_axis.value[~valid], 
                        spectral_axis.value[valid], 
                        spectrum[valid])
                    pah_data[:, i, j] = spectrum
        pah_cube = SpectralCube(data=pah_data*pah_cube.unit, wcs=pah_cube.wcs)
    elif nan_policy == 'ignore':
        # fill NaNs with zeros
        pah_data = pah_cube.unmasked_data[:].value
        pah_data[~np.isfinite(pah_data)] = 0.0
        pah_cube = SpectralCube(data=pah_data*pah_cube.unit, wcs=pah_cube.wcs)
    else:
        raise ValueError(
            f"nan_policy '{nan_policy}' not recognized.")

    # collapse to make PAH image
    chan_width_freq = (
        np.abs(np.median(np.diff(spectral_axis))) *
        const.c / pah_cube.spectral_axis**2).to('Hz')
    pah_image = (
        pah_cube * chan_width_freq.reshape((-1, 1, 1))).sum(axis=0)
    pah_image = pah_image.to('erg s-1 cm-2 sr-1')

    # write to file
    if outfile is not None:
        pah_image.write(outfile, overwrite=overwrite)

    return pah_image
