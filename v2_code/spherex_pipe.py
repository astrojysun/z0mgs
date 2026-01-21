import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from utils_spherex import (
    write_template_tab,
    search_spherex_images, download_images,
    bksub_images, make_cube_header, build_sed_cube, 
    grid_spherex_cube, estimate_continuum_fls,
    make_spherex_line_image, make_spherex_pah_image_naive)

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Set the control flow
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# Set which steps to do by toggling these.

# TBD replace with command line calls.

do_download = True
do_bksub = True
do_sed_cube = True
do_grid = True
do_estcont = True
do_lines = True

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Set parameters
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# Root directory
root_dir = Path('../../test_data/spherex/')

# Target list table and a list to either restrict to and/or skip
#targ_tab = 'targets_spherex.ecsv'
targ_tab = 'spherex_phangs_targets.ecsv'
just_targs = []
skip_targs = []

# Define flags to use
flags_to_use = [
    'SUR_ERROR', 'NONFUNC', 'MISSING_DATA',
    'HOT', 'COLD', 'NONLINEAR', 'PERSIST']

# Wavelengths of spectral features to flag. This assumes features in
# the frame of the target galaxy (assuming MW is handled by background
# subtraction).

features_to_flag = [
    'PAH3.3+3.4',
    'Paa', 'Pab', 'Pag', 'Bra', 'Brb', 'Brg', 'Pfb', 'Pfg',
    # (ad-hoc) He airglow and aurora and CO? features
    'HeI1.083', 'aurora1.65', 'CO?2.175',
    ]
feature_dict = {}
feature_dict['PAH3.3+3.4'] = {'lam':3.35*u.um, 'width':0.2*u.um}
# https://www.gemini.edu/observing/resources/near-ir-resources/spectroscopy/hydrogen-recombination-lines
feature_dict['Paa'] = {'lam':1.87561*u.um, 'width':0.0*u.um}
feature_dict['Pab'] = {'lam':1.28216*u.um, 'width':0.0*u.um}  # auroral contamination...
feature_dict['Pag'] = {'lam':1.09411*u.um, 'width':0.0*u.um}  # He airglow contamination...
feature_dict['Bra'] = {'lam':4.05226*u.um, 'width':0.0*u.um}
feature_dict['Brb'] = {'lam':2.62587*u.um, 'width':0.0*u.um}  # close to band 2-3 edges...
feature_dict['Brg'] = {'lam':2.16612*u.um, 'width':0.0*u.um}  # CO bandhead contamination...
feature_dict['Pfb'] = {'lam':4.65378*u.um, 'width':0.0*u.um}
feature_dict['Pfg'] = {'lam':3.74056*u.um, 'width':0.0*u.um}
# (ad-hoc) He airglow and aurora and CO? features
feature_dict['HeI1.083'] = {'lam':1.083*u.um, 'width':0.05*u.um}
feature_dict['aurora1.65'] = {'lam':1.65*u.um, 'width':0.15*u.um}
feature_dict['CO?2.175'] = {'lam':2.175*u.um, 'width':0.125*u.um}

# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
# Handle the targets
# $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

# Write a template table (gives an example of what to modify)
write_template_tab()

# write_sample_tab(tags=['PHANGS'], outfile='spherex_phangs_targets.ecsv', just_gals=['ngc1808'])
skip_targs = ['m31', 'm33']

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
        this_vrad = 0.0*u.km/u.s

    this_vwidth = this_row['vwidth']
    if not np.isfinite(this_vwidth):
        this_vwidth = 0.0*u.km/u.s        

    # Set directories for this target
    gal_dir = root_dir / this_gal
    raw_ext = 'raw/'
    bksub_ext = 'bksub/'
    raw_dir = gal_dir / raw_ext
    bksub_dir = gal_dir / bksub_ext
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

        gal_dir.mkdir(exist_ok=True)
        raw_dir.mkdir(exist_ok=True)
        
        # Query IRSA to get the list of images

        image_tab = search_spherex_images(
            coordinates=this_coord, radius=this_fov,
            collection='spherex_qr2', verbose=True)

        # Download or check for existence of images

        for i in range(2):
            print(f"\n=====\nTry {i+1}\n=====\n")
            downloaded_images = download_images(
                image_tab, outdir=str(raw_dir)+'/', alt_dirs=alt_dirs,
                incremental=True, verbose=True)

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Background subtract the raw images
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_bksub:

        print("")
        print("Background subtracting and masking the images.")
        print("")

        # Create the background subtracted directory
        
        bksub_dir.mkdir(exist_ok=True)

        # Find the level 2 images

        lvl2_im_list = [
            str(f) for f in raw_dir.glob('level2_*.fits')]
        
        # Background subtract then write out result as a file
        bksub_im_list = bksub_images(
            image_list=lvl2_im_list,
            indir_ext=raw_ext, outdir_ext=bksub_ext,
            sub_zodi=True,
            gal_coord=this_coord,
            gal_rad_deg=this_mask_rad.to(u.deg).value,
            gal_incl=this_mask_incl.to(u.deg).value,
            gal_pa=this_mask_pa.to(u.deg).value,
            frac_bw_step=0.5)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Make an ungridded "SED cube"
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    

    if do_sed_cube:

        print("")
        print("Building irregularly sampled SED+lam+bw cubes.")
        print("")

        im_list = [
            str(f) for f in bksub_dir.glob('bksub*.fits')]
        
        # Make an empty HDU with header only
        cube_hdu = make_cube_header(
            center_coord=this_coord,
            pix_scale=(3.*u.arcsec).to(u.deg).value,
            extent=this_fov.to(u.deg).value, 
            lam_min=0, lam_max=len(im_list), lam_step=1,
            return_header=False)

        # Build the ungridded SED cube
        build_sed_cube(
            target_hdu=cube_hdu, image_list=im_list,
            ext_to_use='BKSUB',
            outfile=str(gal_dir / (this_gal+'_spherex_seds.fits')),
            overwrite=True)
    
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Grid into a cube
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
    
    if do_grid:

        print("")
        print("Building regularly gridded spectral cubes.")
        print("")

        # Build the regularly gridded spectral cube
        grid_spherex_cube(
            int_cube=str(gal_dir / (this_gal+'_spherex_seds.fits')),
            lam_min=0.7, lam_max=5.2, lam_step=0.0075, lam_unit='um',
            outfile=str(gal_dir / (this_gal+'_spherex_cube.fits')),
            method='TOPHAT', overwrite=True)

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&
    # Estimate a continuum from the cube
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_estcont:
    
        print("")
        print("Estimating continuum.")
        print("")

        estimate_continuum_fls(
            int_cube=gal_dir / (this_gal+'_spherex_seds.fits'),
            features_to_flag=features_to_flag, feature_dict=feature_dict,
            vrad=this_vrad, vwidth=this_vwidth,
            lam_min=0.7, lam_max=5.2, lam_step=0.0075, lam_unit='um',
            filter_width=0.2*u.um, bandwidth_fraction=0.15,
            outfile_cube=gal_dir / (this_gal+'_spherex_cube_smooth.fits'),
            outfile_seds=gal_dir / (this_gal+'_spherex_seds_smooth.fits'),
            overwrite=True, verbose=True)

    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&    
    # Make line images
    # $&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&$&

    if do_lines:
    
        print("")
        print("Making line images.")
        print("")

        for line in ['Paa', 'Pab', 'Bra', 'Brb', 'Pfb']:

            print(f"...{line}")

            make_spherex_line_image(
                int_cube=gal_dir / (this_gal+'_spherex_seds.fits'),
                cont_cube=gal_dir / (this_gal+'_spherex_seds_smooth.fits'),
                feature_dict=feature_dict[line],
                vrad=this_vrad, lsf='tophat',
                outfile=gal_dir / (this_gal+f'_spherex_line_{line}_tophat.fits'),
                overwrite=True)

            make_spherex_line_image(
                int_cube=gal_dir / (this_gal+'_spherex_seds.fits'),
                cont_cube=gal_dir / (this_gal+'_spherex_seds_smooth.fits'),
                feature_dict=feature_dict[line],
                vrad=this_vrad, lsf='gaussian',
                outfile=gal_dir / (this_gal+f'_spherex_line_{line}_gaussian.fits'),
                overwrite=True)

        print("...PAH")

        make_spherex_pah_image_naive(
            int_cube=gal_dir / (this_gal+'_spherex_cube.fits'),
            cont_cube=gal_dir / (this_gal+'_spherex_cube_smooth.fits'),
            lam_min=3.1*u.um, lam_max=3.6*u.um, nan_policy='interp',
            outfile=gal_dir / (this_gal+'_spherex_line_PAH_naive.fits'),
            overwrite=True)