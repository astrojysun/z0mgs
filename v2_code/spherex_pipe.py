# Temporary SPHEREx pipeline

import os, glob
from utils_spherex import *
import astropy.units as u
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Set the control flow
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# Set which steps to do by toggling these.

# TBD replace with command line calls.

do_download = False
do_bksub = False
do_sed_cube = True
do_grid = True
do_estcont = True
do_lines = False

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Set parameters
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# Root directory
root_dir = '../../test_data/spherex/'

# Target list table and a list to either restrict to and/or skip
targ_tab = 'targets_spherex.ecsv'
just_targs = ['evan_cloud']
skip_targs = []

# Define the flags to apply
flags_to_use = \
    ['SUR_ERROR','NONFUNC','MISSING_DATA',
     'HOT','COLD','NONLINEAR','PERSIST']

# Wavelengths of spectral features to flag. This assumes features in
# the frame of the target galaxy (assuming MW is handled by background
# subtraction).

features_to_flag = ['pa','pb','bra','brb']
feature_dict = {}
feature_dict['pa'] = {'lam':1.87561*u.um, 'width': 0.0*u.um}
feature_dict['pb'] = {'lam':1.28216*u.um, 'width': 0.0*u.um}
feature_dict['bra'] = {'lam':4.05226*u.um, 'width': 0.0*u.um}
feature_dict['brb'] = {'lam':2.62587*u.um, 'width': 0.0*u.um}

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Handle the targets
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# Write a template table (gives an example of what to modify)
write_template_tab()

# Read the actual table
targ_tab = QTable.read(targ_tab, format='ascii.ecsv')

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Loop over targets
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

for this_row in targ_tab:

    if len(just_targs) > 0:
        if this_row['gal'].strip() not in just_targs:
            continue

    if len(skip_targs) > 0:
        if this_row['gal'].strip() in skip_targs:
            continue

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Get parameters for this target
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    # Set some reasonable defaults and trap missing data.
    
    this_gal = this_row['gal']

    this_ra = this_row['ra']
    this_dec = this_row['dec']

    if (not np.isfinite(this_ra)) or \
       (not np.isfinite(this_dec)):
        this_coord = SkyCoord.from_name(this_gal)
    else:
        this_coord = SkyCoord(ra=this_ra, dec=this_dec, frame='icrs')

    this_fov = this_row['fov']
    if not np.isfinite(this_fov):
        this_fov = 10.*u.arcmin

    this_mask_rad = this_row['mask_rad']
    if not np.isfinite(this_mask_rad):
        this_mask_rad = this_fov

    this_mask_pa = this_row['mask_pa']
    this_mask_incl = this_row['mask_incl']    
    if (not np.isfinite(this_mask_pa)) or \
       (not np.isfinite(this_mask_incl)):
        this_mask_pa = 0.0*u.deg
        this_mask_incl = 0.0*u.deg

    this_vrad = this_row['vrad']
    if not np.isfinite(this_vrad):
        this_vrat = 0.0*u.km/u.s

    this_vwidth = this_row['vwidth']
    if not np.isfinite(this_vwidth):
        this_vwidth = 0.0*u.km/u.s        
        
    # Set directories for this target
    
    gal_dir = root_dir + this_gal + '/'
    raw_ext = 'raw/'
    bksub_ext = 'bksub/'
    raw_dir = gal_dir + raw_ext
    bksub_dir = gal_dir + bksub_ext
    alt_dirs = ['../../test_data/spherex/*/raw/']

    # Print what we're doing

    print("Target: ", this_gal)
    print("Output directory: ", gal_dir)
    print("Coords: ", this_coord)
    print("Field of view: ", this_fov)
    print("Mask radius, incl, pa: ",
          this_mask_rad, this_mask_incl, this_mask_pa)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Download the data
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
    
    if do_download:

        print("")
        print("Querying the archive and downloading data.")
        print("")
        
        # Make the directories if needed
        
        if os.path.isdir(gal_dir) == False:
            os.system('mkdir '+gal_dir)

        if os.path.isdir(raw_dir) == False:
            os.system('mkdir '+raw_dir)

        # Query IRSA to get the list of images
            
        image_tab = \
            search_spherex_images(
                #target = this_gal,
                coordinates = this_coord,
                radius = this_fov,
                collection = 'spherex_qr2',
                verbose = True)

        # Download or check for existence of images
        
        downloaded_images = \
            download_images(
                image_tab,
                outdir = raw_dir,
                alt_dirs = alt_dirs,
                incremental = True,
                verbose = True)

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Background subtract the raw images
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
        
    if do_bksub:

        print("")
        print("Background subtracting and masking the images.")
        print("")
        
        # Create the background subtracted directory
        
        if os.path.isdir(bksub_dir) == False:
            os.system('mkdir '+bksub_dir)

        # Find the level 2 images
            
        lvl2_im_list = glob.glob(raw_dir+'level2_*.fits')
    
        # Background subtract then write out result as a file
        
        bksub_im_list = \
            bksub_images(
                image_list = lvl2_im_list,
                indir_ext = raw_ext,
                outdir_ext = bksub_ext,
                sub_zodi = True,
                gal_coord = this_coord,
                gal_rad_deg = this_mask_rad.to(u.deg).value,
                gal_incl = this_mask_incl.to(u.deg).value,
                gal_pa = this_mask_pa.to(u.deg).value,
                frac_bw_step = 0.5,
            )

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Make an ungridded "SED cube"
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    

    if do_sed_cube:

        print("")
        print("Building irregular-sampling SED+lam+bw cubes.")
        print("")

        im_list = glob.glob(bksub_dir+'bksub*.fits')
        n_images = len(im_list)

        # Make a header that has our desired astrometry and a
        # wavelength axis that is just equal to the number of
        # images. The images are reprojected and loaded in to form
        # cubes of intensity, wavelength, and bandwidth.
        
        cube_hdu = make_cube_header(
            center_coord = this_coord,
            pix_scale = 3. / 3600.,
            extent = this_fov.to(u.deg).value, 
            lam_min = 0, lam_max = n_images, lam_step = 1.0,
            return_header=False)

        build_sed_cube(
            target_hdu = cube_hdu,
            image_list = im_list,
            ext_to_use = 'BKSUB',
            outfile = gal_dir + this_gal + '_spherex_seds.fits',
            overwrite=True)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Grid into a cube with regular wavelength
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
        
    if do_grid:

        # Grid the SED cubes into cubes regularly spaced in
        # wavelength.
        
        grid_spherex_cube(
            int_cube = gal_dir + this_gal + '_spherex_seds.fits',
            lam_min = 0.7, lam_max = 5.2, lam_step = 0.0075,
            lam_unit = 'um',            
            outfile = gal_dir + this_gal+'_spherex_cube.fits',
            method = 'TOPHAT',
            overwrite = True)

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Estimate a continuum from the cube
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_estcont:

        estimate_continuum(
            int_cube = gal_dir + this_gal + '_spherex_seds.fits',
            lam_min = 0.7, lam_max = 5.2, lam_step = 0.0075, lam_unit = 'um',
            features_to_flag = features_to_flag,
            feature_dict = feature_dict,
            vrad = this_vrad, vwidth = this_vwidth,
            outfile_cube = gal_dir + this_gal+'_spherex_cube_smooth.fits',
            outfile_seds = gal_dir + this_gal+'_spherex_sed_smooth.fits',            
            overwrite=True)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
    # Make line images
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_lines:
        
        make_spherex_line_image(
            int_cube = gal_dir + this_gal + '_spherex_seds.fits',
            cont_cube = gal_dir + this_gal + '_spherex_seds_smooth.fits'
            feature_dict = feature_dict['bra'],
            vrad=this_vrad, lsf='tophat',
            outfile=gal_dir + this_gal + '_spherex_line_bra_tophat.fits',
            overwrite = True)
        
        make_spherex_line_image(
            int_cube = gal_dir + this_gal + '_spherex_seds.fits',
            cont_cube = gal_dir + this_gal + '_spherex_seds_smooth.fits'
            feature_dict = feature_dict['bra'],
            vrad=this_vrad, lsf='gaussian',
            outfile=gal_dir + this_gal + '_spherex_line_bra_gaussian.fits',
            overwrite = True)
        
