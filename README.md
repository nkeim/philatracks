# philatracks

Python modules for combining particle tracking of jammed 2D packings with oscillatory rheology.

Written by [Nathan Keim](http://www.seas.upenn.edu/~nkeim/), a member of the [Penn Complex Fluids Lab](http://arratia.seas.upenn.edu).

Portions of this code were used in data analysis for
    Keim, N. C. & Arratia, P. E. Yielding and microstructure in a 2D jammed material under shear deformation. *Soft Matter* **9**, 6222 (2013).

Unit tests may be run with `py.test`.

## Installation

`philatracks` makes heavy use of the fabulous [`pandas`](http://pandas.pydata.org/) library for analysis of structured data. It and other more standard requirements (`numpy`, `scipy`) are found in most scientific Python distributions.

To install:

    pip install 'git+http://github.com/nkeim/philatracks/#egg=philatracks'

For obtaining and reading tracks data, see [`runtrackpy`](https://github.com/nkeim/runtrackpy/)


