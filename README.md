# 4f-Correlator
Simulation code in Python for a 4f Correlator 

This repository contains a simple simulation for the optical processes in a 4f correlator using Fourier optics. Given an input image (the location of which must be specified in the code), it will calculate the Fourier transform of said image and the Fourier transform after filtering certain spatial frequencies. The code allows for simple filters such as horizontal and vertical straight lines, as well as circular filters centered at the origin. The wavelength of the light used as the source can be specified but it is assumed to be monochromatic. The 4f correlator being modeles has 2 thin lenses of equal focus and separated by 2f.
