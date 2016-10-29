#include "loadMNIST.H"
#include <iostream>
#include <string>
#include <fstream>

using namespace arma;

void loadMNIST(mat& X, vec& y){

  const std::string filenames[] = {"./MNISTdata/data0", "./MNISTdata/data1", "./MNISTdata/data2", 
				   "./MNISTdata/data3", "./MNISTdata/data4", "./MNISTdata/data5", 
				   "./MNISTdata/data6", "./MNISTdata/data7", "./MNISTdata/data8", "./MNISTdata/data9"};
  
  const size_t N = sizeof(filenames)/sizeof(filenames[0]); // string filenames[0] same length as all others
  size_t m = X.n_rows/10;
  size_t n = X.n_cols;	
      
  for (size_t k = 0; k != N; ++k)
  {
    std::ifstream infile(filenames[k].c_str(), std::ios::in|std::ios::binary|std::ios::ate); // open binary file for input and set cursor to end
    
    if (infile.is_open())
    {
      std::ifstream::pos_type size;
      size = infile.tellg(); // current cursor position (at end due to ios::ate)
      char* memblock; 
      int num;
      memblock = new char[size]; 
      infile.seekg(0, std::ios::beg); // cursor back to beginning
      infile.read(memblock, size); // stores file into memory

      for (size_t i = 0; i != m; ++i)
      {
	for (size_t j = 0; j != n; ++j)
	{
	  num = (int)memblock[i*n + j]; 
	  /* convert to binary */
	  if (num != 0)
	  {
	    X(m*k + i, j) = 1;
	  }
	  y(m*k+i) = k;
	}
      }

      infile.close();
      delete[] memblock;
    }
    else 
    {
      std::cout << "unable to open file " << filenames[k] << std::endl;
    }
  }
  
  /* code to shuffle */

  mat ymat = conv_to<mat>::from(y); 

  mat Xy = join_rows(X, ymat);
  Xy = shuffle(Xy);
  X = Xy.cols(0, n-1); // n-1 is penultimate col of Xy
  ymat = Xy.col(n);

  y = conv_to<vec>::from(ymat);
}
