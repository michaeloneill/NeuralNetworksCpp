#include "NNet.H"
#include "loadMNIST.H"
#include "matrixToFile.H"
#include "Tuning.H"


#include <armadillo>
#include <iostream>
#include <sstream>

#include <vector>

#include "Layers.H" // for plotting basis functions
#define TUNING


using namespace arma;
using std::vector;

int main() {


    /* load data */

    uword nSamples = 10000;
    uword dimInput = 784;
    mat X = zeros(nSamples, dimInput);
    vec y = zeros(nSamples);
    loadMNIST(X, y); // already shuffled



    /* define model parameters */
    
    size_t epochs = 30;
    size_t miniBatchSize = 20;
    double alpha = 0.1;
    double lambda = 0;
    size_t nlabels = 10;

    
    /* write 100 images to output file for plotting */

    mat X_tofile = X.rows(0, 99);
    matrixToFile(X_tofile, "outputDigits");
    


    /* split into train and test */

    size_t nTrain = 1000;
    mat X_train = X.rows(0, nTrain-1); 
    vec y_train = y.rows(0, nTrain-1);
    mat X_test = X.rows(nTrain, nSamples-1);
    vec y_test = y.rows(nTrain, nSamples-1);

// REMEMBER TO CHANGE MACRO OPTION BELOW TOO - TO DELETE CORRECT NUMBER OF LAYER POINTERS    
#if 1

    /* Case 1: Single hidden layer fully connected, with fully connected output layer */

    uword dimHidden = 25; // 100 performs better than 25
    
    vector<Layer*> layers;
 
    Layer* layer1 = new FullyConnectedLayer(dimInput, dimHidden);
    Layer* layer2 = new FullyConnectedLayer(dimHidden, nlabels);

    layers.push_back(layer1);
    layers.push_back(layer2);

    
#elif 0
    
    /* Case 2: Single hidden layer fully connected, with Softmax output layer */

    uword dimHidden = 25;
    
    vector<Layer*> layers;
 
    Layer* layer1 = new FullyConnectedLayer(dimInput, dimHidden);
    Layer* layer2 = new SoftmaxLayer(dimHidden, nlabels);

    layers.push_back(layer1);
    layers.push_back(layer2);



#elif 0

    /* Case 3: Two hidden layers: fully connected, fully connected, softmax  output */

    uword dimHidden1 = 25;
    uword dimHidden2 = 100;
    
    vector<Layer*> layers;
 
    Layer* layer1 = new FullyConnectedLayer(dimInput, dimHidden1);
    Layer* layer2 = new FullyConnectedLayer(dimHidden1, dimHidden2);
    Layer* layer3 = new SoftmaxLayer(dimHidden2, nlabels);

    layers.push_back(layer1);
    layers.push_back(layer2);
    layers.push_back(layer3);



#elif 0

    /* Case 4: Single hidden layer ConvPool, with Softmax output layer */

    uword imRows = 28;
    uword imCols = 28;
    uword numFilters = 20;
    uword filterDim = 5;
    uword poolDim = 2;
    uword softIn = numFilters*(imRows-filterDim+1)*(imCols-filterDim+1)/(poolDim*poolDim);
    uword softOut = nlabels;
   
    vector<Layer*> layers;
 
    Layer* layer1 = new ConvPoolLayer(filterDim, numFilters, poolDim, imRows, imCols);
    Layer* layer2 = new SoftmaxLayer(softIn, softOut);

    layers.push_back(layer1);
    layers.push_back(layer2);



#elif 0

    /* Case 5: Two hidden layer - ConvPool, FullyConnected, Softmax */
    
    uword imRows = 28;
    uword imCols = 28;
    uword numFilters = 20;
    uword filterDim = 5;
    uword poolDim = 2;
    uword FullyIn = numFilters*(imRows-filterDim+1)*(imCols-filterDim+1)/double(poolDim*poolDim);
    uword FullyOut = 25;
   
    vector<Layer*> layers;
 
    Layer* layer1 = new ConvPoolLayer(filterDim, numFilters, poolDim, imRows, imCols);
    Layer* layer2 = new FullyConnectedLayer(FullyIn, FullyOut);
    Layer* layer3 = new SoftmaxLayer(FullyOut, nlabels);

    layers.push_back(layer1);
    layers.push_back(layer2);
    layers.push_back(layer3);
    

///////////////////////// CASE 6 IS FOR FUTURE WORK AND TAKES > 10hrs to train ////////////////////////
    
#elif 0

    /* Case 6: Three hidden layers - ConvPool, convPool, FullyConnected, Softmax */
    
    uword imRows1 = 10;
    uword imCols1 = 10;
    uword imRows2 = 4;
    uword imCols2 = 16;
    uword numFilters1 = 4;
    uword numFilters2 = 2;
    uword filterDim = 3;
    uword poolDim = 2;

    uword FullyIn = numFilters2*(imRows2-filterDim+1)*(imCols2-filterDim+1)/double(poolDim*poolDim);
    uword FullyOut = 100;
   
    vector<Layer*> layers;
 
    Layer* layer1 = new ConvPoolLayer(filterDim, numFilters1, poolDim, imRows1, imCols1);
    Layer* layer2 = new ConvPoolLayer(filterDim, numFilters2, poolDim, imRows2, imCols2);
    Layer* layer3 = new FullyConnectedLayer(FullyIn, FullyOut);
    Layer* layer4 = new SoftmaxLayer(FullyOut, nlabels);

    layers.push_back(layer1);
    layers.push_back(layer2);
    layers.push_back(layer3);
    layers.push_back(layer4);
    

    
#endif
    

    /* Initialise, train and test classifier */

    NNet clf(layers);

    clf.train(X_train, y_train, epochs, miniBatchSize, alpha, lambda, true); // monitor cost

    vec predictions = clf.predict(X_test);
    double accuracyTrain = clf.score(X_train, y_train);
    double accuracyTest = clf.score(X_test, y_test);
    std::cout << "accuracy on training set is "<< accuracyTrain << std::endl;
    std::cout << "accuracy on test set is "<< accuracyTest << std::endl;
  

    /* output learned basis functions */
    
    typedef vector<Layer*>::const_iterator layerIter;
    for (layerIter iter = layers.begin(); iter != layers.end(); ++iter)
    {
    	std::stringstream ss;
    	ss << "outputNNBasisLayer" << iter-layers.begin()+1;
    	std::string filename = ss.str();
    	(*iter)->basisOut(filename); 
    }
    
   

#ifdef TUNING

    /* output learning curve data to file for plotting */

    try
    {
    	learningCurves(clf, X_train, y_train, epochs, miniBatchSize, alpha, lambda, linspace<vec>(0.1, 1.0, 100), 0.8); 
    }
    catch (std::domain_error e)
    {
    	std::cout << e.what() << std::endl;
    }


    /* output cross validation data to file for plotting */

    //lambda
    try
    {
    	validationCurves(clf, X_train, y_train, epochs, miniBatchSize, alpha, linspace<vec>(0, 20, 21), 0.8, "lambda"); 
    }
    catch (std::domain_error e)
    {
    	std::cout << e.what() << std::endl;
    }

    
    // alpha
    try
    {
    	validationCurves(clf, X_train, y_train, epochs, miniBatchSize, lambda, linspace<vec>(0, 4, 41), 0.8, "alpha"); 
    }
    catch (std::domain_error e)
    {
    	std::cout << e.what() << std::endl;
    }

    
#endif


#if 1 // for cases 1, 2 and 4
    delete layer1;
    delete layer2;
#elif 0 // for cases 3 and 5
    delete layer1;
    delete layer2;
    delete layer3;
#elif 0 // for case 6 
    delete layer1;
    delete layer2;
    delete layer3;
    delete layer4;
#endif
    
    
    return 0;

}









