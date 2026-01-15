# Temporary SPHEREx pipeline

import os, glob
from utils_spherex import *
import astropy.units as u

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Define targets
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# For now use a dictionary of galaxy name and image size. We will
# expand this eventually.

gal_list = {
    'ngc5194':20.*u.arcmin,
}

# Define the flags to use

flags_to_use = \
    ['SUR_ERROR','NONFUNC','MISSING_DATA',
     'HOT','COLD','NONLINEAR','PERSIST']

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Set the control flow
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

do_download = False
do_bksub = False
do_sed_cube = True
do_grid = False
do_estcont = False
do_lines = False

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Loop over targets
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

for this_gal, this_rad in gal_list.items():

    # Relevant directories
    
    root_dir = '../../test_data/spherex/'
    gal_dir = root_dir + this_gal + '/'
    raw_dir = gal_dir + 'raw/'
    bksub_dir = gal_dir + 'bksub/'
    alt_dirs = ['../../test_data/spherex/*/raw/']

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Download the data
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
    
    if do_download:

        # Make the directories if needed
        
        if os.path.isdir(gal_dir) == False:
            os.system('mkdir '+gal_dir)

        if os.path.isdir(raw_dir) == False:
            os.system('mkdir '+raw_dir)

        # Query IRSA to get the list of images
            
        image_tab = \
            search_spherex_images(
                target = this_gal,
                radius = this_rad,
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

        # Create the background subtracted directory
        
        if os.path.isdir(bksub_dir) == False:
            os.system('mkdir '+bksub_dir)

        # Find the level 2 images
            
        lvl2_im_list = glob.glob(raw_dir+'level2_*.fits')
    
        # Background subtract then write out result as a file
        
        bksub_im_list = \
            bksub_images(
                image_list = lvl2_im_list,
                indir_ext = 'raw/',
                outdir_ext = 'bksub/',
                sub_zodi = True,
                gal_coord = None,
                gal_rad_deg = 10./60.,
                gal_incl = 0.0,
                gal_pa = 0.0,
                frac_bw_step = 0.5,
            )

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Make an ungridded "SED cube"
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    

    if do_sed_cube:
    
        im_list = glob.glob(bksub_dir+'bksub*.fits')
        n_images = len(im_list)

        cube_hdu = make_cube_header(
            center_coord = this_gal,
            #pix_scale = 6. / 3600.,
            pix_scale = 3. / 3600.,
            extent = this_rad.to(u.deg).value, 
            lam_min = 0, lam_max = n_images, lam_step = 1.0,
            return_header=False)

        build_sed_cube(
            target_hdu = cube_hdu,
            image_list = im_list,
            flags_to_use = flags_to_use,
            ext_to_use = 'IMAGE',
            outfile = gal_dir + this_gal + '_spherex_seds.fits',
            overwrite=True)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Grid into a cube
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
        
    if do_grid:
    
        cube_hdu = make_cube_header(
            center_coord = this_gal,
            #pix_scale = 6. / 3600.,
            pix_scale = 3. / 3600.,
            extent = this_rad.to(u.deg).value, 
            lam_min = 0.75, lam_max = 5.2, lam_step = 0.0075,
            return_header=False)

        im_list = glob.glob(bksub_dir+'bksub*.fits')

        grid_spherex_cube(
            target_hdu = cube_hdu,
            image_list = im_list,
            ext_to_use = 'BKSUB',            
            outfile = gal_dir + this_gal+'_spherex_cube.fits',
            flags_to_use = flags_to_use,
            overwrite = True)

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Estimate a continuum from the cube
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_estcont:

        estimate_continuum(
            cube = gal_dir + this_gal+'_spherex_cube.fits',
            features_to_flag = {
                1.875:0.04,
                4.052:0.04,
            },
            outfile = gal_dir + this_gal+'_spherex_smooth.fits',
            overwrite=True)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
    # Make line images
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_lines:
        
        spherex_line_image(
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
            overwrite = True)
        
    
