#ifndef TUNING_H
#define TUNING_H

#include "BaseClassifier.H"
#include <armadillo>

#include <string>

void learningCurves(BaseClassifier&, arma::mat&, arma::vec&, size_t, size_t, double, double, const arma::vec, const double);

void validationCurves(BaseClassifier&, arma::mat&, arma::vec&, size_t, size_t, double, const arma::vec, const double, std::string);

#endif
