#include "matrixToFile.H"
#include <fstream>

using namespace arma;

void matrixToFile(const mat& X, const std::string file){

    std::ofstream outfile;
    outfile.open(file.c_str());

    if (outfile.is_open())
    {
	size_t m = X.n_rows;
	size_t n = X.n_cols;

	for (size_t i = 0; i != m; ++i)
	{
	    for (size_t j = 0; j != n; ++j)
	    {
		outfile << X(i, j) << " ";
	    }
	    outfile << std::endl;
	}
	outfile.close();
    }
    else
    {
	std::cout << "unable to open file: " << file << std::endl;
    }
}
