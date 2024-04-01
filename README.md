# TESSELLATE: TESS Extensive Lightcurve Logging and Analysis of Transient Events

This pipeline utilises the MAST archive and TESSreduce (see https://github.com/CheerfulUser/TESSreduce) to conduct an untargeted transient event search through TESS data.

Designed for SLURM usage on the OzStar supercomputer at Swinburne University.

To install this package, just run `pip install git+https://github.com/rhoxu/TESSELLATE.git`. This will download the core package, and also a custom TESSreduce package modified to operate on the supercomputer. I recommend actually cloning this to your local drive so that changes can be made easily when bugs are inevitibly found (there are loads).

Have a look at the example notebook `example.ipynb`. This gives a step by step explanation for how to go about accessing data and creating new tessellation runs. Note that *__200 sec cadence sectors currently are not operational__*.

Again, this code likely will be buggy, and also there are many cases where jobs simply fail on the supercomputer, and then work immediately afterwards for no apparent reason. I suggest *__running each CCD indiviudally__* rather than iterating over cameras and ccds, as this generally seems more successful and is easier to track. 

If you have any questions whatsoever, flick me (Hugh) an email at roxburghhugh@gmail.com

