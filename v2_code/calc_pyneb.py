import pyneb as pn

# figure out the theoretical Pa9/Ha line ratio

# make H I recombination atom
H1 = pn.RecAtom('H', 1)

# choose conditions
Te = 10000.0   # K
#Te = 3500.0   # K
ne = 100.0     # cm^-3

# Halpha: 3->2, wavelength ~6562.8 A
# Pa9: transition 9->3 -> wavelength ~9231.6 A (check exact lambda in your line list)
lambda_Ha = 6562.8
lambda_Hb = 4862.3
lambda_Pa9 = 9231.6

# get emissivities (units: erg cm^3 s^-1) for case B
eps_Ha = H1.getEmissivity(tem=Te, den=ne, wave=lambda_Ha)#, case='B')
eps_Hb = H1.getEmissivity(tem=Te, den=ne, wave=lambda_Hb)#, case='B')
eps_Pa9 = H1.getEmissivity(tem=Te, den=ne, wave=lambda_Pa9)#, case='B')

ratio = eps_Pa9 / eps_Ha
print(f"I(Pa9)/I(Ha) (Case B) at Te={Te} K, ne={ne} cm^-3  = {ratio:.4e}")
ratio_hahb = eps_Ha/eps_Hb
print(f"I(Ha)/I(Hb) (Case B) at Te={Te} K, ne={ne} cm^-3  = {ratio_hahb:.4e}")
