# Coursework for MPHYGB07
[MPHYGB07](http://www.ucl.ac.uk/medphys/prospective-students/modules/mphygb07) is the *Computer-assisted Surgery and Therapy* course at University College London, whose lecturer is [Dr. Tom Vercauteren](http://iris.ucl.ac.uk/iris/browse/profile?upi=TVERC65).

This coursework is an analysis of [Deep Retinal Image Understanding (DRIU)](http://www.vision.ee.ethz.ch/~cvlsegmentation/driu/). The repository includes the LaTeX code of the report and the Python code written to run the experiments on the [DRIVE](https://www.isi.uu.nl/Research/Databases/DRIVE/) dataset and generate the corresponding figures.

## Installation
```shell
# Optionally create a conda environment...
conda create -n cast python=3
# ...and activate it
source activate cast

# Download and install everything
pip install -e git+https://github.com/fepegar/cast-coursework.git#egg=cast
```
