# TESSELLATE: TESS Extensive Lightcurve Logging and Analysis of Transient Events

This pipeline utilises the MAST archive and TESSreduce (see https://github.com/CheerfulUser/TESSreduce) to conduct an untargeted transient event search through TESS data.

Designed for SLURM usage on the OzStar supercomputer at Swinburne University.

To get started, download this repository and navigate into it via a command terminal. Make sure you are in the home folder containing setup.py, and then run `pip install .` to install the package. Have a look at the example notebook `example.ipynb`.

The `TESSreduce-master` folder contains changes to the regular TESSreduce package so that it can be used on the supercomputer - just navigate into the `TESSreduce-master` directory and run `pip install .`.

If you have any questions whatsoever, flick me (Hugh) an email at roxburghhugh@gmail.com

To come: updates about detection systems...

