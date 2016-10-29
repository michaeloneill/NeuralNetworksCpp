The source code files in miniProject2.tar.gz relate to the mini-project report "Multi-layer neural networks applied to the classification of hand-written digits".

The MNIST data files should be downloaded from http://cis.jhu.edu/~sachin/digit/digit.html and organised by digit into a folder MNISTdata, with MNISTdata/data0 corresponding to digit 0 etc. all the way up to digit 9.

The project uses the c++ armadillo linear algebra library. The included Makefile has the required linker flags to this library for running the code on the LSC computers. 

The results for each of the 5 cases in the aforementioned report can be re-produced by

1) toggling on\off the macro conditionals in main.C that refer to each case (and changing parameters, which are labelled with self-explanatory names).
2) togglng the corresponding macro conditionals at the end of main.C to delete the appropriate number of layer pointers.
3) updating the case variable in plotNNet.py to match that used in main.C.
4) Note that if the hidden layer dimensions for any of the cases are changed in main.C, the dimension of the subplot grids in plotNNet.py must be changed accordingly, else plotNNEt.py will not be able to generate the learned basis plots (the rest of the plots will work fine). 

To compile the program:

make NNet

To run it:

./NNet

Numerical gradient checking can be performed by toggling on/off the CHECK_GRADIENTS macro in NNet.C. 

A mentioned above, the file plotNNet.py can be used to generate all the plots. Note that output data files for the plotting of learning curves and validation curves will only be produced if the TUNING macro in main.C is set to true.


