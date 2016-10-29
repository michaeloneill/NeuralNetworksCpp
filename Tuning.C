#include "Tuning.H"
#include "matrixToFile.H"

/* Polymorphism enabled by passing by reference rather than pointer */
/* This means function can't accept NULL objects */

using namespace arma;
using std::string;

void learningCurves(BaseClassifier& clf, mat& X, vec& y, size_t epochs, size_t miniBatchSize, double alpha, double lambda, const vec learningBatches, const double split){

    std::cout << "computing Learning curve data " << std::endl << std::flush;
    uword N = X.n_rows;
    uword nLearningBatches = learningBatches.n_elem;

    vec score_train = zeros(nLearningBatches);
    vec score_val = zeros(nLearningBatches);

    if (split <= 1.0 && split > 0)
    {
	/* Use some of training data for validation */
	
	uword nTrain = ceil(split*N);
	mat X_train = X.rows(0, nTrain-1);
	vec y_train = y.rows(0, nTrain-1);
	mat X_val = X.rows(nTrain, N-1);
	vec y_val = y.rows(nTrain, N-1);
	
	for (uword i = 0; i != nLearningBatches; ++i)
	{
	    std::cout << "Learning Curves: Training batch " << i+1 << std::endl;
	    
	    if (learningBatches(i) <= 1.0 && learningBatches(i) > 0)
	    {
		uword learningBatchSize = ceil(learningBatches(i)*X_train.n_rows);
		mat X_learningBatch = X_train.rows(0, learningBatchSize-1);
		vec y_learningBatch = y_train.rows(0, learningBatchSize-1);
	   
		clf.train(X_learningBatch, y_learningBatch, epochs, miniBatchSize, alpha, lambda, false);
		score_train(i) = clf.score(X_learningBatch, y_learningBatch);
		score_val(i) = clf.score(X_val, y_val);

	    }
	    else
	    {
		throw std::domain_error("learningBatches elements must be (0, 1]");

	    }
	}
    }
    else
    {
	throw std::domain_error("split must be (0, 1]");
    }
    
    mat scores = join_horiz(score_train, score_val);
    scores = join_horiz(conv_to<mat>::from(learningBatches), scores);
    matrixToFile(scores, "outputLC");
    std::cout << std::endl;
}

/* Polymorphism enabled by passing by reference rather than pointer */
/* This means function can't accept NULL objects */


void validationCurves(BaseClassifier& clf, mat& X, vec& y, size_t epochs, size_t miniBatchSize, double fixed, const vec values, const double split, string valType){


    if ((valType != "lambda") && (valType != "alpha"))
    {
	throw std::domain_error("Invalid validation parameter type");
    }		


    std::cout << "computing validation data " << std::endl << std::flush;
    uword N = X.n_rows;
    uword nValues = values.n_elem;

    vec score_train = zeros(nValues);
    vec score_val = zeros(nValues);

    if (split <= 1.0 && split > 0)
    {
	/* Use some of training data for validation */
	
	uword nTrain = ceil(split*N);
	mat X_train = X.rows(0, nTrain-1);
	vec y_train = y.rows(0, nTrain-1);
	mat X_val = X.rows(nTrain, N-1);
	vec y_val = y.rows(nTrain, N-1);
	
	for (uword i = 0; i != nValues; ++i)
	{

	    if (values(i) >= 0)
	    {
		if (valType == "lambda")
		{
		    std::cout << "Lambda Validation Curves: Training for lambda = " << values(i) << std::endl;
		    clf.train(X_train, y_train, epochs, miniBatchSize, fixed, values(i), false);
		}
		else // must be alpha
		{
		    std::cout << "Alpha Validation Curves: Training for alpha = " << values(i) << std::endl;
		    clf.train(X_train, y_train, epochs, miniBatchSize, values(i), fixed, false);
		}

		score_train(i) = clf.score(X_train, y_train);
		score_val(i) = clf.score(X_val, y_val);
	    }
	    else
	    {
		throw std::domain_error("values must be >= 0");

	    }
	}
    }
   
    else
    {
	throw std::domain_error("split must be (0, 1]");
    }

    mat scores = join_horiz(score_train, score_val);
    scores = join_horiz(conv_to<mat>::from(values), scores);
    if (valType == "lambda")
    {
	matrixToFile(scores, "outputLamVal");
    }
    else //must be alpha
    {
	matrixToFile(scores, "outputAlphaVal");
    }
    std::cout << std::endl;
}
