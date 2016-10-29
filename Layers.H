#ifndef LAYERS_H
#define LAYERS_H

#include <armadillo>
#include <vector>
#include <string>
#include "Stack.H"


class Layer {
    friend class NNet; 
public:
    
    Layer() {} //default constructor
    virtual ~Layer(){} // calls appropraite default destructor 
  
    const arma::vec& getParams() const {return mParams;}
    virtual size_t getDimIn() const = 0; //return depend on layer type
    virtual size_t getDimOut() const = 0; 
    virtual size_t getNumWeights() const = 0;
    virtual size_t getNumBiases() const = 0;
   
    virtual arma::mat feedForward(arma::mat&)=0;
    virtual arma::mat feedForward(arma::mat&, const arma::vec&)=0;
    virtual arma::mat backpropDelta(const arma::mat&)=0;
    virtual arma::mat computeDeltas(const arma::mat&, const arma::mat&)=0;
    virtual arma::vec computeGrad(const arma::mat&, const arma::mat&, const arma::mat&)=0;
    virtual arma::vec computeGradFromDelta(const arma::mat&, const arma::mat&)=0;
    virtual double costFn(const arma::mat&, const arma::umat&)=0;
    virtual arma::mat delta(const arma::mat&, const arma::umat&)=0;
    virtual void basisOut(const std::string&)=0;


protected:
    
    arma::vec mParams;
    virtual void initParams()=0;   
};



class FullyConnectedLayer: public Layer {
public:

    FullyConnectedLayer(): mDimIn(0), mDimOut(0){} // default
    FullyConnectedLayer(size_t dimIn, size_t dimOut): mDimIn(dimIn), mDimOut(dimOut){}
    size_t getDimIn() const {return mDimIn;}
    size_t getDimOut() const {return mDimOut;}
    size_t getNumWeights() const {return mDimOut*mDimIn;}
    size_t getNumBiases() const {return mDimOut;} 
   

    arma::mat feedForward(arma::mat&);
    arma::mat feedForward(arma::mat&, const arma::vec&);
    arma::mat backpropDelta(const arma::mat&);
    arma::mat computeDeltas(const arma::mat&, const arma::mat&);
    arma::vec computeGrad(const arma::mat&, const arma::mat&, const arma::mat&);
    arma::vec computeGradFromDelta(const arma::mat&, const arma::mat&);
    double costFn(const arma::mat&, const arma::umat&);
    arma::mat delta(const arma::mat&, const arma::umat&);
    void basisOut(const std::string&);
    
    Stack<arma::mat, arma::vec> paramsToStack(const arma::vec& unrolled);
    
protected:

    void initParams();
    const size_t mDimIn;
    const size_t mDimOut;
   
};



class SoftmaxLayer: public FullyConnectedLayer {

public:

    SoftmaxLayer(){} // uses default base constructor to initialse base part 
    SoftmaxLayer(size_t dimIn, size_t dimOut): FullyConnectedLayer(dimIn, dimOut) {} // uses specified base constructor to initialse base part
    arma::mat feedForward(arma::mat&);
    arma::mat feedForward(arma::mat&, const arma::vec&);
    double costFn(const arma::mat&, const arma::umat&);
    arma::mat delta(const arma::mat&, const arma::umat&);
};



class ConvPoolLayer: public Layer {
public:

    ConvPoolLayer():
	mFilterDim(0)
	, mNumFilters(0)
	, mPoolDim(0)
	, mImRows(0)
	, mImCols(0){}
    
    ConvPoolLayer(arma::uword filterDim, arma::uword numFilters, arma::uword poolDim, arma::uword imRows, arma::uword imCols):
	mFilterDim(filterDim)
	, mNumFilters(numFilters)
	, mPoolDim(poolDim)
	, mImRows(imRows)
	, mImCols(imCols){}


    size_t getDimIn() const {return mImRows*mImCols;}; 
    size_t getDimOut() const {return mNumFilters*(mImRows-mFilterDim+1)*(mImCols-mFilterDim+1)/(mPoolDim*mPoolDim);} 
    
    size_t getNumWeights() const {return mFilterDim*mFilterDim*mNumFilters;}
    size_t getNumBiases() const {return mNumFilters;} 


    arma::mat feedForward(arma::mat&);
    arma::mat feedForward(arma::mat&, const arma::vec&);
    arma::mat backpropDelta(const arma::mat&);
    arma::mat computeDeltas(const arma::mat&, const arma::mat&);
    arma::vec computeGrad(const arma::mat&, const arma::mat&, const arma::mat&);
    arma::vec computeGradFromDelta(const arma::mat&, const arma::mat&);
    double costFn(const arma::mat&, const arma::umat&);
    arma::mat delta(const arma::mat&, const arma::umat&);

    arma::cube convolve(const arma::cube&, arma::cube&, const arma::vec&); 
    arma::cube pool(const arma::cube&);

    void basisOut(const std::string&);
    
    Stack<arma::cube, arma::vec>  paramsToStack(const arma::vec&);

private:
    
    const arma::uword mFilterDim;
    const arma::uword mNumFilters;
    const arma::uword mPoolDim;
    const arma::uword mImRows;
    const arma::uword mImCols;
    
    void initParams();
    arma::cube upsample(const arma::cube&);
};


// non-members

arma::mat rot90(const arma::mat&);
arma::mat myConvolve(const arma::mat&, const arma::mat&);
arma::mat meanPool(const arma::mat&, arma::uword);




#endif
