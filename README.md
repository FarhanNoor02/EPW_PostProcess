# EPW_PostProcess
A collection of scripts to post-process the output from Quantum ESPRESSO's EPW code


#Capabilities:
- aniso_gap.plt : a gnuplot script which read the files ncn.imag_aniso_gap0_03.00 and so on and plot the distribution of anisotropic gap function at each Temperature. Users can change the range of temperatures considered and the initial prefix
- aniso_gap.py: same as above, but as a python script. Adds a color gradient to the plot. Yrange and file prefixes are modified according to user needs 
- qdos.py: reads all the files "ncn.qdos_*.00" in a directory, applies mirroring and plots the superconducting DOS (qdos) as a functon of frequency in a format that is similar to plots used by experimentalists.
- FS_gap.py: reads a file like filename = "ncn.imag_aniso_gap_FS_003.00" and plots Fermi Surface and colors it with corresponding anisotropic superconducting gap function. Asks the user to enter the viewing angles (theta and phi) to tilt the projection plane on which the Fermi surface is projected.
