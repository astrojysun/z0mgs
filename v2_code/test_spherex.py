import os, glob
from utils_spherex import *

gal_list = ['m31']
#gal_list = ['m33']
#gal_list = ['ic10', 'ic1613', 'ngc6822']
#gal_list = ['DDO221'] # this is WLM

#gal_list = ['ngc0253','ngc0300','ngc4594','ngc5194','ngc5236','ngc7793','m33']
#gal_list = ['ngc3034']
#gal_list = ['ngc5194']

re_download = True
make_sed = True
grid_cube = True

for this_gal in gal_list:

    root_dir = '../../test_data/spherex/'
    gal_dir = root_dir + this_gal + '/'
    outdir = gal_dir + 'raw/'
    
    if re_download:
        
        os.system('mkdir '+gal_dir)
        os.system('mkdir '+outdir)
        
        image_tab = \
            search_spherex_images(
                target = this_gal,
                radius = 60*u.arcmin,
                collection = 'spherex_qr2',
                verbose = True)

        downloaded_images = \
            download_images(
                image_tab,
                outdir = outdir,
                incremental = True,
                verbose = True)

    if make_sed:

        im_list = glob.glob(outdir+'*.fits')
    
        extract_spherex_sed(
            target_coord = this_gal,        
            image_list = im_list,
            outfile = gal_dir + this_gal + '_spherex_ctrsed.ecsv',
            overwrite=True)
    
    if grid_cube:
    
        cube_hdu = make_cube_header(
            center_coord = this_gal,
            pix_scale = 6. / 3600.,
            extent = 30. / 60., 
            lam_min = 0.75, lam_max = 5.2, lam_step = 0.0075,
            return_header=False)

        im_list = glob.glob(outdir+'*.fits')

        grid_spherex_cube(
            target_hdu = cube_hdu,
            image_list = im_list,
            outfile = gal_dir + this_gal+'_spherex_cube.fits',
            overwrite = True)
