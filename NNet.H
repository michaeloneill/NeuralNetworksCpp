#ifndef NNET_H
#define NNET_H

#include "BaseClassifier.H"
#include "Layers.H"
#include <armadillo>
#include <vector>



class NNet: public BaseClassifier{
public:

    NNet(std::vector<Layer*>&);

    void train(arma::mat&, arma::vec&, size_t, size_t, double, double, bool);
    arma::vec predict(const arma::mat&);
    double score(const arma::mat&, const arma::vec&);
     
    const std::vector<arma::vec> getParams() const;

    typedef std::vector<Layer*>::const_iterator layerIter;
    typedef std::vector<Layer*>::const_reverse_iterator rLayerIter;


private:

    const std::vector<Layer*>  mLayers;
    const std::vector<Layer*>::size_type mNumLayers;
    
    void initParams();
    void batchSGD(const arma::mat&, const arma::vec&, double, double, arma::uword);
    std::vector<arma::vec> grad(const arma:: mat&, const arma::vec&, double);
    std::vector<arma::vec> grad(const arma::mat&, const arma::vec&, double, arma::uword);
    void feedForward(arma::mat&);
    void feedForward(arma::mat&, const std::vector<arma::vec>&);
    double computeTotalCost(const arma::mat&, const arma::vec&, double);
    
    double computeTotalCost(const arma::mat&, const arma::vec&, const std::vector<arma::vec>&, double);
    void checkGradients(double);
    arma::vec unrollNetParams(const std::vector<arma::vec>&);
    std::vector<arma::vec> stackNetParams(const arma::vec&);
    
};


// non-members
void shuffleData(arma::mat&, arma::vec&); 
arma::umat yToFull(const arma::vec&, size_t);


#endif
