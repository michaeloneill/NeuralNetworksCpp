#include "Layers.H"
#include "ActivationFns.H"
#include "matrixToFile.H"
#include <cassert>
#include <stdexcept>

using namespace arma;
using std::vector;
using std::string;

//#define GOODWEIGHTS


// FullyConnectedLayer members

void FullyConnectedLayer::initParams(){

#ifdef GOODWEIGHTS
    double epsilon = 4*sqrt(6/double(mDimOut + mDimIn));
#else
    double epsilon = 0.12;
#endif

    mat weights = randu(mDimOut, mDimIn)*2*epsilon - epsilon;
    vec biases = zeros(mDimOut);
    mParams = join_vert(vectorise(weights), biases);
}



mat FullyConnectedLayer::feedForward(mat& activations){

    return feedForward(activations, mParams);
}



mat FullyConnectedLayer::feedForward(mat& activations, const vec& params){

    /* batch feedforwards activations from prev layer and returns zs for backprop */
    /* activations is dimPrev*batchSize */

    Stack<mat, vec> paramStack = paramsToStack(params);
    mat temp = paramStack.w*activations;
    mat zs = temp.each_col() + paramStack.b;
    activations = sigmoid(zs);

    return zs;
}



arma::mat FullyConnectedLayer::backpropDelta(const arma::mat& dCurr){

    /* batch back-propagates deltas from this layer to errors in layer before */
    /* dCurr is dimCurr*batchSize */

    Stack<mat, vec> paramStack = paramsToStack(mParams);
    arma::mat errPrev = paramStack.w.t()*dCurr;

    return errPrev;

}



arma::mat FullyConnectedLayer::computeDeltas(const arma::mat& errCurr, const arma::mat& zCurr){
    
    /* batch computes current layer deltas from current layer errors and zs */
    /* both errCurr and zCurr are dimCurr*batchSize */
    
    return errCurr%sigmoidGradient(zCurr);

}



vec FullyConnectedLayer::computeGrad(const arma::mat& errCurr, const arma::mat& zCurr, const arma::mat& aPrev){

    /* batch Compute dCurr from errCurr by chain rule before computing grads */

//mat dCurr = errCurr%sigmoidGradient(zCurr);

    mat dCurr = computeDeltas(errCurr, zCurr);
    
    return computeGradFromDelta(dCurr, aPrev);
}



vec FullyConnectedLayer::computeGradFromDelta(const arma::mat& dCurr, const arma::mat& aPrev){

    /* require this version if this is final layer. In this case we use cost deltas rather than cost errors, as the former are in general easier to compute and lead to improved stability */
    /* batch computes aggregate gradient w.r.t the parameters in this layer */
    /* returns unrolled gradients w.r.t all weights and biases */ 
    /* un-normalised and un-regularised */
    /* dCurr is dimCurr*batchSize */
    /* aPrev is dimPrev*batchSize */

    arma::mat gradW = dCurr*aPrev.t(); // result is dimCurr*dimPrev - sums up gradient contributions from each sample in batch 
    arma::vec gradb= sum(dCurr, 1); // dimension dimCurr - sums up gradient contributions from each sample in batch

    return join_vert(vectorise(gradW), gradb);

}



double FullyConnectedLayer::costFn(const mat& activations, const umat& yFull){

    /* returns unregularised, un-normailsed batch cost */
    /* activations is nlabels*nSamples representing output from final layer. */
    /* yFull is nlabels*nSamples boolean matrix */ 

    mat YFull = conv_to<mat>::from(yFull); //to allow negation
    double cost = sum(sum(-YFull%log(activations) - (1-YFull)%log(1-activations)));    

    return cost;
}



mat FullyConnectedLayer::delta(const mat& activations, const umat& yFull){

    /* derivative of cost w.r.t final layer neuron inputs (zs). */
    /* each column is delta due to a different sample */
    mat Y = conv_to<mat>::from(yFull); //for negation
    return activations-Y;
}


void FullyConnectedLayer::basisOut(const string& filename)
{
    // Call after training to output the learned basis for plotting

    Stack<mat, vec> paramStack = paramsToStack(mParams);
    matrixToFile(paramStack.w, filename);
}



Stack<mat, vec> FullyConnectedLayer::paramsToStack(const vec& unrolled){

    uword curPos = 0;
    
    uword wLen = mDimOut*mDimIn;
    mat weights = unrolled.rows(curPos, curPos+wLen-1);
    weights.reshape(mDimOut, mDimIn);
    curPos = curPos + wLen;

    uword bLen = mDimOut;
    vec biases = unrolled.rows(curPos, curPos+bLen-1);
    curPos = curPos + bLen;

    Stack<mat, vec> paramStack(weights, biases);
			
    return paramStack;
}



mat SoftmaxLayer::feedForward(mat& activations){

    return feedForward(activations, mParams);
}



mat SoftmaxLayer::feedForward(mat& activations, const vec& params){

    
    Stack<mat, vec> paramStack = paramsToStack(params);
    uword nSamples = activations.n_cols;
   
    mat temp = paramStack.w*activations;

    mat zs = temp.each_col() + paramStack.b;
    activations = exp(zs); // before normalisation of columns

    for (uword i = 0; i != nSamples; ++i)
    {
	activations.col(i) = activations.col(i)/accu(activations.col(i)); // normalised

    }

    return zs;
}



double SoftmaxLayer::costFn(const mat& probs, const umat& yFull){
    
    double cost = -sum(sum(log(probs)%yFull));
  
    return cost;
  
}



mat SoftmaxLayer::delta(const mat& probs, const umat& yFull){

    return probs-yFull;
}





// ConvPool layer members


cube ConvPoolLayer::pool(const cube& aConv){

    /* transforms convolved activations for each image and filter into corresponding pooled activations */
    /* As always, this is done by sample batch */

    uword numImages = aConv.n_slices/mNumFilters;
    uword resRows = (mImRows-mFilterDim+1)/mPoolDim;
    uword resCols = (mImCols-mFilterDim+1)/mPoolDim;

    cube aPooled = zeros(resRows, resCols, numImages*mNumFilters);

    for (uword imageNum = 0; imageNum != numImages; ++imageNum)
    {
	for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
	{
	    mat convolution = aConv.slice(imageNum*mNumFilters + filterNum);
	    mat pooled = meanPool(convolution, mPoolDim);
	    aPooled.slice(imageNum*mNumFilters+filterNum) = pooled;
	}
    }
    return aPooled;
}



cube ConvPoolLayer::convolve(const cube& images, cube& aConv, const vec& params){

   /* updates aConv (passed by reference) and returns convolved zs for backprop */

    uword numImages = images.n_slices; //batch size
    Stack<cube, vec> paramStack = paramsToStack(params);

    uword convRows = (mImRows-mFilterDim+1);
    uword convCols = (mImCols-mFilterDim+1);

    // need these for backprop 
    cube zConv = zeros(convRows, convCols, numImages*mNumFilters);

    for (uword imageNum = 0; imageNum != numImages; ++imageNum)
    {
	mat image = images.slice(imageNum); //image to convolve each filter over

	for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
	{
	    mat filter = paramStack.w.slice(filterNum);
	    
	    // conv of given image with given filter 
	    mat convolution = myConvolve(image, filter);

	    //add bias for filter (element-wise) and apply activation
	    mat zs = convolution + paramStack.b(filterNum);
	    convolution = sigmoid(zs);

	    aConv.slice(imageNum*mNumFilters + filterNum) = convolution;
	    zConv.slice(imageNum*mNumFilters + filterNum) = zs;
	}
    }
    return zConv;
}



void ConvPoolLayer::initParams(){

#ifdef GOODWEIGHTS
    double epsilon = 4*sqrt(6/double(getDimIn() + getDimOut()));
#else
    double epsilon = 0.12;
#endif
    cube weights = randu(mFilterDim, mFilterDim, mNumFilters)*2*epsilon - epsilon;
    vec biases = zeros(mNumFilters);
    
    weights.reshape(mFilterDim*mFilterDim*mNumFilters, 1, 1);
    mParams = join_vert(vectorise(weights.slice(0)), biases);
}



mat ConvPoolLayer::feedForward(mat& activations){

    return feedForward(activations, mParams);
}



mat ConvPoolLayer::feedForward(mat& activations, const vec& params){

    /* note activations passed by reference, so value updated */
    
    assert(activations.n_rows == mImRows*mImCols);
    
    uword numImages = activations.n_cols;

    // reshape activations into 'images' to convolve over
    cube images(mImRows*mImCols, numImages, 1);
    images.slice(0) = activations;
    images.set_size(mImRows, mImCols, numImages);

    // initialse storage for convolved activations
    uword convRows = mImRows-mFilterDim+1;
    uword convCols = mImCols-mFilterDim+1;
    cube aConv = zeros(convRows, convCols, numImages*mNumFilters);

    cube zConv = convolve(images, aConv, params); // note this changes aConv
    cube aPooled = pool(aConv);

    uword resRows = convRows/mPoolDim;
    uword resCols = convCols/mPoolDim;

    //stack units as columns for each image
    aPooled.set_size(resRows*resCols*mNumFilters, numImages, 1);
    zConv.set_size(convRows*convCols*mNumFilters, numImages, 1);

    activations = aPooled.slice(0); 

    return zConv.slice(0); // for backprop
}



arma::mat ConvPoolLayer::computeDeltas(const mat& errPooledUnrolled, const mat& zConvUnrolled){

    /* upsamples pooled errors and returns unrolled convolved deltas */
    /* dimCurr*batchSize is 'unrolled form" */
   /* errPooledUnrolled and zConvUnrolled are both in this form too */
    

    uword numImages = errPooledUnrolled.n_cols;
    uword convRows = mImRows-mFilterDim+1;
    uword convCols = mImCols-mFilterDim+1;

    uword resRows = convRows/mPoolDim;
    uword resCols = convCols/mPoolDim;
    
    cube errPooled = zeros(resRows*resCols*mNumFilters, numImages, 1); 
    errPooled.slice(0) = errPooledUnrolled;
    errPooled.set_size(resRows, resCols, numImages*mNumFilters);

    cube errConv = upsample(errPooled);

    errConv.set_size(convRows*convCols*mNumFilters, numImages, 1);

    mat dConvUnrolled = errConv.slice(0)%sigmoidGradient(zConvUnrolled);
	
    return dConvUnrolled;

}



arma::mat ConvPoolLayer::backpropDelta(const mat& dConvUnrolled){
  
    /* batch back-propagates deltas from conv layer to errors in the layer before */


    uword numImages = dConvUnrolled.n_cols; //batch size

    uword convRows = mImRows-mFilterDim+1;
    uword convCols = mImCols-mFilterDim+1;

    
    cube dConv = zeros(convRows*convCols*mNumFilters, numImages, 1);
    dConv.slice(0) = dConvUnrolled;
    dConv.set_size(convRows, convCols, numImages*mNumFilters);

   
    // now backprop dConv to the layer before
     
    cube errPrev = zeros(mImRows, mImCols, numImages);    
   
    Stack<cube, vec> paramStack = paramsToStack(mParams);

    // aggregating over filters
    for (uword imageNum = 0; imageNum != numImages; ++imageNum)
    {	
	for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
	{
	    mat ds = dConv.slice(imageNum*mNumFilters+filterNum);

            // we must pad ds so that myConvole will perform a 'full' convolution  
	    mat dsPadded = zeros(mImRows+mFilterDim-1, mImCols+mFilterDim-1);
	    dsPadded.submat(mFilterDim-1, mFilterDim-1, mImRows-1, mImCols-1) = ds;

	    mat filter = paramStack.w.slice(filterNum);
	    filter = rot90(rot90(filter)); // see backprop maths

	    errPrev.slice(imageNum) += myConvolve(dsPadded, filter);
	}	
    }

    errPrev.set_size(mImRows*mImCols, numImages, 1);
    return errPrev.slice(0);
}



vec ConvPoolLayer::computeGrad(const mat& errPooledUnrolled, const mat& zConvUnrolled, const mat& aPrev){
    
    /* returns unrolled gradients w.r.t all parameters */
    /* First upsamples pooled errors to convolved errors */
    /* and converts these to convolved deltas via chain rule */  
    
    uword numImages = errPooledUnrolled.n_cols; // the batch size
    
    uword resRows = (mImRows-mFilterDim+1)/mPoolDim;
    uword resCols = (mImCols-mFilterDim+1)/mPoolDim;
    cube errPooled = zeros(resRows*resCols*mNumFilters, numImages, 1);
    errPooled.slice(0) = errPooledUnrolled;

    //upsample
    errPooled.set_size(resRows, resCols, numImages*mNumFilters);
    cube errConv = upsample(errPooled);

    uword convRows = mImRows-mFilterDim+1;
    uword convCols = mImCols-mFilterDim+1;

    errConv.set_size(convRows*convCols*mNumFilters, numImages, 1);

    //convert to deltas
    cube dConv = errConv;
    dConv.slice(0) = errConv.slice(0)%sigmoidGradient(zConvUnrolled);

    dConv.set_size(convRows, convCols, numImages*mNumFilters);


    // reshape aPrev into 'images' for working with
    cube images = zeros(mImRows*mImCols, numImages, 1);
    images.slice(0) = aPrev;
    images.set_size(mImRows, mImCols, numImages);

    
    // compute gradW and gradb by aggragating over images, then return unrolled
    cube gradW = zeros(mFilterDim, mFilterDim, mNumFilters);
    vec gradb = zeros(mNumFilters);
    vec gradUnrolled;

    for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
    {
	for (uword imageNum = 0; imageNum != numImages; ++imageNum)
	{
	    gradW.slice(filterNum) += myConvolve(images.slice(imageNum), dConv.slice(imageNum*mNumFilters + filterNum));

	    gradb(filterNum) += accu(dConv.slice(imageNum*mNumFilters + filterNum));
	}

	gradUnrolled = join_vert(gradUnrolled, vectorise(gradW.slice(filterNum)));
    }
    return join_vert(gradUnrolled, gradb);
}



vec ConvPoolLayer::computeGradFromDelta(const mat& dPooledUnrolled, const mat& aPrev){


    /* require this version if this is final layer. In this case we use cost deltas rather than cost errors, as the former are in general easier to compute and lead to improved stability */


    uword numImages = dPooledUnrolled.n_cols; // the batch size
    
    uword resRows = (mImRows-mFilterDim+1)/mPoolDim;
    uword resCols = (mImCols-mFilterDim+1)/mPoolDim;
    cube dPooled = zeros(resRows*resCols*mNumFilters, numImages, 1);
    dPooled.slice(0) = dPooledUnrolled;

    //upsample
    dPooled.set_size(resRows, resCols, numImages*mNumFilters);
    cube dConv = upsample(dPooled);

    // reshape aPrev for working with
    cube images = zeros(mImRows*mImCols, numImages, 1);
    images.slice(0) = aPrev;
    images.set_size(mImRows, mImCols, numImages);

    
    // compute gradW and gradb by aggregating over images, then return unrolled
    cube gradW = zeros(mFilterDim, mFilterDim, mNumFilters);
    vec gradb = zeros(mNumFilters);
    vec gradUnrolled;

    for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
    {
	for (uword imageNum = 0; imageNum != numImages; ++imageNum)
	{
	    gradW.slice(filterNum) += myConvolve(images.slice(imageNum), dConv.slice(imageNum*mNumFilters + filterNum));

	    gradb(filterNum) += accu(dConv.slice(imageNum*mNumFilters + filterNum));
	}

	gradUnrolled = join_vert(gradUnrolled, vectorise(gradW.slice(filterNum)));
    }
    return join_vert(gradUnrolled, gradb);
}



void ConvPoolLayer::basisOut(const string& filename)
{
    // Call after training to output the learned basis for plotting

    std::ofstream outfile;
    Stack<cube, vec> paramStack = paramsToStack(mParams);

    outfile.open(filename.c_str(), ios::out|ios::trunc);

    if (outfile.is_open())
    {
	for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
	{
	    for (uword i = 0; i != mFilterDim; ++i)
	    {
		for (uword j = 0; j != mFilterDim; ++j)
		{
		    outfile << paramStack.w.slice(filterNum)(i, j) << " ";
		}
		outfile << std::endl;
	    }
	}
        outfile.close();
    }
    else
    {
	std::cout << "file not open: " << filename << std::endl;
    }
}



Stack<cube, vec> ConvPoolLayer::paramsToStack(const vec& unrolled){

    uword curPos = 0;
    cube weights = zeros(mFilterDim, mFilterDim, mNumFilters);
    uword filterLen = mFilterDim*mFilterDim;

    for (uword i = 0; i != mNumFilters; ++i)
    {
	mat filter = unrolled.rows(curPos, curPos+filterLen-1);
	filter.reshape(mFilterDim, mFilterDim);
	weights.slice(i) = filter;
	curPos = curPos + filterLen;
    }
    
    uword bLen = mNumFilters;
    vec biases = unrolled.rows(curPos, curPos+bLen-1);
    curPos = curPos + bLen;
    
    Stack<cube, vec> paramStack(weights, biases);
			
    return paramStack;
}

cube ConvPoolLayer::upsample(const cube& dPooled){

    uword numImages = dPooled.n_slices/mNumFilters; // batch size
    uword convRows = mImRows-mFilterDim+1;
    uword convCols = mImCols-mFilterDim+1;
    cube dConv = zeros(convRows, convCols, numImages*mNumFilters);

    for (uword imageNum = 0; imageNum != numImages; ++imageNum)
    {
	for (uword filterNum = 0; filterNum != mNumFilters; ++filterNum)
	{
	    mat ds = dPooled.slice(imageNum*mNumFilters+filterNum);
	    mat dsUpsampled = kron(ds, ones(mPoolDim, mPoolDim)/double(mPoolDim*mPoolDim));
	    dConv.slice(imageNum*mNumFilters+filterNum) = dsUpsampled;
	}
    }
    return dConv;
}



double ConvPoolLayer::costFn(const mat& activations, const umat& yFull){

    /* Uses FullyConnectedLayer XEntropy */
    /* activations would be post pooling */
    /* Note: Highly unusual to make ConvPool final layer */
    /* If you did you would need to ensure dimension of activations was nLabels */
    
    mat YFull = conv_to<mat>::from(yFull); //to allow negation
    double cost = sum(sum(-YFull%log(activations) - (1-YFull)%log(1-activations)));    

    return cost;
}



mat ConvPoolLayer::delta(const mat& activations, const umat& yFull){


    /* Uses FullyConnectedLayer XEntropy */
    /* activations would be post pooling */
    /* Note: Highly unusual to make ConvPool final layer */
    /* If you did you would need to ensure dimension of activations was nLabels */

    
    mat Y = conv_to<mat>::from(yFull); //for negation
    return activations-Y;
}





//non-members


mat rot90(const mat& matrix){
    
    /* mimics matlab's function (counter-clockwise) */

    return flipud(trans(matrix));
}

mat myConvolve(const mat& matrix, const mat& filter){
    
    /* Performs a 'valid' convolve. */
    /* 'full convole achieved by passing padded matrix to this */

    uword filterRows = filter.n_rows;
    uword filterCols = filter.n_cols;
    uword matRows = matrix.n_rows;
    uword matCols = matrix.n_cols;

    assert(filterRows <= matRows && filterCols <= matCols);
    
    uword convRows = matRows-filterRows+1;
    uword convCols = matCols-filterCols+1;   

    mat convolution = zeros(convRows, convCols);

    for (uword i = 0; i != convRows; ++i)
    {
    	for (uword j = 0; j != convCols; ++j)
    	{
    	    mat patch = matrix.submat(i, j, i+filterRows-1, j+filterCols-1);
    	    convolution(i, j) = accu(filter%patch);
    	}
    }
    return convolution;
}

mat meanPool(const mat& matrix, uword poolDim){

// poolDim must be a factor of matDim

    uword matRows = matrix.n_rows;
    uword matCols = matrix.n_cols;
    
    assert((matRows%poolDim == 0) && (matCols%poolDim == 0));
    
    uword resRows = matRows/poolDim;
    uword resCols = matCols/poolDim;
    
    mat pooled = zeros(resRows, resCols); 

    for (uword pooledRow = 0; pooledRow != resRows; ++pooledRow)
    {
	uword rowBegin = pooledRow*poolDim;
	uword rowEnd = rowBegin+poolDim-1;

	for (uword pooledCol = 0; pooledCol != resCols; ++pooledCol)
	{
	    uword colBegin = pooledCol*poolDim;
	    uword colEnd = colBegin+poolDim-1;
	    mat patch = matrix.submat(rowBegin, colBegin, rowEnd, colEnd);
	    pooled(pooledRow, pooledCol) = mean(vectorise(patch));
	}
    }
    
    return pooled;
}




