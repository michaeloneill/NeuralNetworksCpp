#include "NNet.H"
#include "matrixToFile.H"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cassert>


#define CHECK_GRADIENTS

using namespace arma;
using std::vector;


//public members

NNet::NNet(vector<Layer*>& layers): mLayers(layers), mNumLayers(layers.size()) {

    for (layerIter iter = layers.begin(); iter != layers.end()-1; ++iter)
    {
	assert((*iter)->getDimOut() == (*(iter+1))->getDimIn());
    }
#ifdef CHECK_GRADIENTS
    double lambda = 0;
    initParams(); 
    checkGradients(lambda);
#endif

}



void NNet::train(mat& X_train, vec& y_train, size_t epochs, size_t miniBatchSize, double alpha, double lambda, bool trackCost){

    initParams();
    uword nTrain = X_train.n_rows;
    assert(y_train.n_rows == nTrain);
    size_t nBatches = ceil(nTrain/(double)miniBatchSize);
    std::ofstream outfile;
    std::string filename = "outputNNCostHistory";

    if (trackCost)
    {
	outfile.open(filename.c_str(), ios::out|ios::trunc);
    }    

    for (size_t i = 0; i != epochs; ++i)
    {
	shuffleData(X_train, y_train);
	for (size_t j = 0; j != nBatches; ++j)
	{
	    uword first = j*miniBatchSize;
	    uword last = (j==nBatches-1)? nTrain-1: first+miniBatchSize-1;
	    mat XBatch = X_train.rows(first, last);
	    vec yBatch = y_train.rows(first, last);
	    batchSGD(XBatch, yBatch, alpha, lambda, nTrain);
	}
	std::cout << "Epoch " << i+1 << " training complete" << std::endl;
	
	if (outfile.is_open())
	{
	    outfile << i+1 << " " << computeTotalCost(X_train, y_train, lambda) << std::endl;
	}
	else
	{
	    if (trackCost)
	    {
		std::cout << "file not open: " << filename << std::endl;	
	    }
	}	
    }
    outfile.close();
}



vec NNet::predict(const mat& X_test){

    assert(X_test.n_cols == mLayers.front()->getDimIn());

    size_t nSamples = X_test.n_rows;
    uvec predictions = zeros<uvec>(nSamples); // must use uvec for max

    mat activations = X_test.t();	
    feedForward(activations);
	
    //index of max in col of activations is class prediction for corresponding sample
    for (uword i = 0; i != nSamples; ++i)
    {
	activations.col(i).max(predictions(i));
    }
    return conv_to<vec>::from(predictions); 
}



double NNet::score(const mat& X, const vec& y){

    uword nSamples = X.n_rows;
    assert(nSamples == y.n_rows);
    
    vec predictions = predict(X);
    return accu(predictions == y)/(double)nSamples;    
}



const vector<vec> NNet::getParams() const {

    vector<vec> netParams;
    for (layerIter iter = mLayers.begin(); iter != mLayers.end(); ++iter)
    {
	vec layerParams = (*iter)->getParams();
	netParams.push_back(layerParams);
    }
    return netParams;
}




// private members 

void NNet::initParams(){

    for (layerIter iter = mLayers.begin(); iter != mLayers.end(); ++iter)
    {
	(*iter)->initParams();
    }
}



void NNet::batchSGD(const mat& XBatch, const vec& yBatch, double alpha, double lambda, uword nTrain){

    // note returned batchGrad is already normailsed and regularised
    vector<vec> batchGrad = grad(XBatch, yBatch, lambda, nTrain);
    //SGD
    for (size_t i = 0; i != mNumLayers; ++i)
    {
    	mLayers[i]->mParams = mLayers[i]->mParams - alpha*batchGrad[i];
    }
}



vector<vec> NNet::grad(const mat& X, const vec& y, double lambda){
    
    return grad(X, y, lambda, X.n_rows);

}



vector<vec> NNet::grad(const mat& X, const vec& y, double lambda, uword nTrain){ 
/* simultaneous backpropagation for a batch */ 
/* y is a boolean matrix */
/* activations has samples as columns */

    uword nSamples = X.n_rows;
    vector<vec> batchGrad(mNumLayers); // for storing layer grads
    
    /* feed forward */

    // input layer
    mat activations  = X.t();
    umat yFull = yToFull(y, mLayers.back()->getDimOut());
    
    vector<mat> activationsHistory;
    activationsHistory.push_back(activations);
    vector<mat> zsHistory;

    // inner layers
    for (layerIter iter = mLayers.begin(); iter!= mLayers.end(); ++iter)
    {
	mat zs = (*iter)->feedForward(activations);
	zsHistory.push_back(zs);
	activationsHistory.push_back(activations);
    }

    /* backprop */

    // Note activationsHistory is one element longet than mNumLayers

    // Last layer
    mat errors;
    mat deltas = mLayers.back()->delta(activationsHistory.back(), yFull);
    batchGrad.back() = mLayers[mNumLayers-1]->computeGradFromDelta(deltas, activationsHistory[mNumLayers-1]); 

    // Hidden layers
    for (size_t i = 1; i != mNumLayers; ++i)
    {
	size_t layerIndex = mNumLayers-i;
	errors = mLayers[layerIndex]->backpropDelta(deltas);
	batchGrad[layerIndex-1] = mLayers[layerIndex-1]->computeGrad(errors, zsHistory[layerIndex-1], activationsHistory[layerIndex-1]);
	deltas = mLayers[layerIndex-1]->computeDeltas(errors, zsHistory[layerIndex-1]);
    }

    // normalise and regularise

    vector<vec> netParams = getParams();
    
    for (vector<vec>::size_type i = 0; i != mNumLayers; ++i)
    {

	batchGrad[i] /= (double)nSamples; // normalise

	//regularise

	size_t numWeights = mLayers[i]->getNumWeights();
	size_t numBiases = mLayers[i]->getNumBiases();
	vec weights = netParams[i].rows(0, numWeights-1);
	weights.insert_rows(numWeights, numBiases); // no regularisation applied to bias
	batchGrad[i] += lambda/((double)nTrain)*weights;

    }

    return batchGrad;
}
    


void NNet::feedForward(mat& activations){

    for (layerIter iter = mLayers.begin(); iter != mLayers.end(); ++iter)
    {
	(*iter)->feedForward(activations);	
    }

}    



void NNet::feedForward(mat& activations, const vector<vec>& netParams){


    for (vector<vec>::size_type i = 0; i != mNumLayers; ++i)
    {
	mLayers[i]->feedForward(activations, netParams[i]);	

    }

}    


double NNet::computeTotalCost(const mat& X, const vec& y, double lambda){
    
    // calling without parameter argument implies use stored parameters
    return computeTotalCost(X, y, getParams(), lambda); 
}



double NNet::computeTotalCost(const mat& X, const vec& y, const vector<vec>& netParams, double lambda){

/* computes cost over whole dataset passed in */
    
    uword nSamples = X.n_rows;
    mat activations = X.t(); // cols are samples now
    umat yFull = yToFull(y, mLayers.back()->getDimOut());

    feedForward(activations, netParams);

    double cost = mLayers.back()->costFn(activations, yFull);

    // normalise

    cost /= (double)nSamples;

    // regularise
    
    double l2NrmSqrd = 0; 
	
    for (vector<vec>::size_type i = 0; i != mNumLayers; ++i)
    {
	size_t numWeights = mLayers[i]->getNumWeights();
	vec weights = netParams[i].rows(0, numWeights-1);
	l2NrmSqrd += accu(square(weights));
    }

    cost += lambda/(2*(double)nSamples) * l2NrmSqrd;
    return cost;

}



void NNet::checkGradients(double lambda){ 


    // generate small net

    size_t inputDim = mLayers.front()->getDimIn();
    size_t outputDim = mLayers.back()->getDimOut();
    uword nSamples = 10;
    vec XUnrolled = zeros(nSamples*inputDim);

    for (uword i = 0; i != nSamples*inputDim; ++i)
    {
	XUnrolled(i) = sin(i+1)/10;
    }

    mat X = reshape(XUnrolled, nSamples, inputDim);
    vec y = zeros(nSamples);
    
    for (uword i = 0; i != nSamples; ++i)
    {
	y(i) = (i+1)%outputDim;
    }
    
    // compute numerical gradient
    
    vec netParamsUnrolled = unrollNetParams(getParams());
    uword len = netParamsUnrolled.n_elem;
    vec numGradsUnrolled = zeros(len);
    vec dParams = zeros(len);
    double delta = 1e-4;

    vec linecount(len);
    
    for (uword i = 0; i < len; ++i)
    {
	linecount(i) = i+1;
	dParams(i) = delta;
	vec params1 = (netParamsUnrolled - dParams);
	vec params2 = (netParamsUnrolled + dParams);

	double cost1 = computeTotalCost(X, y, stackNetParams(params1), lambda);
	double cost2 = computeTotalCost(X, y, stackNetParams(params2), lambda);
	numGradsUnrolled(i) = (cost2 - cost1)/(2*delta);
	dParams(i) = 0; // reset for next round
	if ((i+1)%100 == 0)
	{
	    std::cout << (i+1) << " numerical gradients calculated" << std::endl;
	}
    }

    // compare with backprop grad

    vector<vec> netGrads = grad(X, y, lambda);
    vec netGradsUnrolled = unrollNetParams(netGrads);
    std::cout << std::scientific << join_horiz(linecount, join_horiz(numGradsUnrolled, netGradsUnrolled)) << std::endl;


}



vec NNet::unrollNetParams(const vector<vec>& netParams){

    vec unrolled;
    uword curPos = 0;
    
    for (uword i = 0; i != mNumLayers; ++i){

	unrolled.insert_rows(curPos, netParams[i]);
	curPos += netParams[i].n_rows;
    }

    return unrolled;
}



vector<vec> NNet::stackNetParams(const vec& unrolled){

    uword curPos = 0;
    vector<vec> netParams;
    
    for (uword i = 0; i != mNumLayers; ++i)
    {	
	uword paramLen = mLayers[i]->mParams.n_rows;
	vec layerParams = unrolled.rows(curPos, curPos+paramLen-1);
	curPos = curPos + paramLen;
	
	netParams.push_back(layerParams);
			
    }
    
    return netParams;
}



// non-members
			   
void shuffleData(mat& X, vec& y){

    uword dim = X.n_cols;

    mat ymat = conv_to<mat>::from(y); 
    mat Xy = join_rows(X, ymat);
    Xy = shuffle(Xy);
    X = Xy.cols(0, dim-1); // dim-1 is penultimate col of Xy
    ymat = Xy.col(dim);
    y = conv_to<vec>::from(ymat);

}

umat yToFull(const vec& y, size_t nlabels){

    size_t nSamples = y.n_rows;
    umat yFull = zeros<umat>(nlabels, nSamples);
    for (size_t i = 0; i != nSamples; ++i)
    {
	yFull(y(i), i) = 1;
    }
    return yFull;

}


// vector<Stack> NNet::grad(const mat&X, const vec& y, double lambda, uword nTrain){

//     vector<Stack> gradStack;
//     size_t nSamples = X.n_rows;
    
//     for (size_t i = 0; i != mNumLayers; ++i)
//     {
// 	mat wGrad = zeros(size(mLayers[i]->getParams().w));
// 	vec bGrad = zeros(size(mLayers[i]->getParams().b));
// 	gradStack.push_back(Stack(wGrad, bGrad));

//     }    

//     backprop(X, y, gradStack);

//     //normalise over nSamples and add regularisation

//     for (size_t i = 0; i != mNumLayers; ++i)
//     {
//     	gradStack[i].w = gradStack[i].w/((double)nSamples) + lambda/((double)nTrain)*mLayers[i]->getParams().w;
//     	gradStack[i].b = gradStack[i].b/((double)nSamples);
	
//     }

//     return gradStack;
    
// }

