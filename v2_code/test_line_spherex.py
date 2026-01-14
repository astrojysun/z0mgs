import os, glob
from utils_spherex import *

#gal_list = ['ngc0253','ngc0300','ngc4594','ngc5194','ngc5236','ngc7793','m33']
#gal_list = ['ngc3034']
#gal_list = ['m33']
#gal_list = ['ngc5194','m33','ngc0253']
#gal_list = ['m31']
gal_list = ['ngc5194','ngc0253','ngc5236','ngc7793','ngc0300',
            'ic10', 'ic1613', 'DDO221', 'm33']

re_download = False
make_line = True

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
                radius = 20*u.arcmin,
                collection = 'spherex_qr2',
                verbose = True)

        downloaded_images = \
            download_images(
                image_tab,
                outdir = outdir,
                incremental = True,
                verbose = True)
    
    if make_line:
        
        flags_to_use = \
            ['SUR_ERROR','NONFUNC','MISSING_DATA',
             'HOT','COLD','NONLINEAR','PERSIST']
            
        #flags_to_use = \
            #    ['NONFUNC','MISSING_DATA']
    
        target_hdu = make_cube_header(
            center_coord = this_gal,
            pix_scale = 3. / 3600.,
            extent = 20. / 60., 
            lam_min = 0.75, lam_max = 5.2, lam_step = 0.0075,
            return_header=False)

        im_list = glob.glob(outdir+'*.fits')
        
        for this_line in ['bra', 'pa']:            

            if this_line == 'bra':

                off_lo_lam = 3.95
                off_hi_lam = 4.25
                on_lam = 4.05226

            if this_line == 'pa':

                off_lo_lam = 1.77
                off_hi_lam = 1.97
                on_lam = 1.87561
                
            print("Making an off-line image at lower wavelength")
            
            off_low_hdu = spherex_line_image(
                target_hdu = target_hdu,
                central_lam = off_lo_lam,
                vel_width = 6000.,
                frac_thresh = 0.10,
                image_list = im_list,
                operation = 'intensity',
                flags_to_use = flags_to_use,
                sub_bkgrd = True,
                gal_coord = None,
                gal_rad_deg = 10.0/60.0,
                gal_incl = 0.,
                gal_pa = 0.,
                overwrite = True)
            
            print("Making an off-line image at higher wavelength")
        
            off_high_hdu = spherex_line_image(
                target_hdu = target_hdu,
                central_lam = off_hi_lam,
                vel_width = 6000.,
                frac_thresh = 0.10,
                image_list = im_list,
                operation = 'intensity',
                flags_to_use = flags_to_use,
                sub_bkgrd = True,
                gal_coord = None,
                gal_rad_deg = 10.0/60.0,
                gal_incl = 0.,
                gal_pa = 0.,
                overwrite = True)
            
            off_image = 0.5*(off_low_hdu.data + off_high_hdu.data)
            off_hdu = fits.PrimaryHDU(off_image, off_high_hdu.header)
            off_hdu.writeto(gal_dir + this_gal + '_'+this_line+'_off.fits',
                            overwrite=True)

            print("Making the on-line image")
            
            spherex_line_image(
                target_hdu = target_hdu,
                central_lam = on_lam,
                vel_width = 500.,
                frac_thresh = 0.75,
                image_list = im_list,
                continuum = gal_dir + this_gal + '_'+this_line+'_off.fits',
                outfile = gal_dir + this_gal+'_'+this_line+'.fits',
                operation = 'integrate',
                flags_to_use = flags_to_use,
                sub_bkgrd = True,
                gal_coord = None,
                gal_rad_deg = 10.0/60.0,
                gal_incl = 0.,
                gal_pa = 0.,
                overwrite = True)
            
