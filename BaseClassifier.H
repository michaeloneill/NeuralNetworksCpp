#ifndef BASECLASSIFIER_H
#define BASECLASSIFIER_H

#include <armadillo>

class BaseClassifier {
public:

    virtual ~BaseClassifier(){}
    virtual void train(arma::mat&, arma::vec&, size_t, size_t, double, double, bool) = 0;
    virtual arma::vec predict(const arma::mat&) = 0;
    virtual double score(const arma::mat&, const arma::vec&) = 0;
    
    
};

#endif
