#include "ActivationFns.H"

using namespace arma;

mat sigmoid(const mat& z){

    mat h = zeros<mat>(size(z));
    h = 1/(1+exp(-z));
    return h;

}

mat sigmoidGradient(const mat& z){
    
    mat hgrad = zeros<mat>(size(z));
    hgrad = sigmoid(z)%(1-sigmoid(z));
    return hgrad;

}
