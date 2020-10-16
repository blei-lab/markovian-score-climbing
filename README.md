# Markovian Score Climbing: Variational Inference with KL(p||q)

This Git-repository includes source code to run experiments for
```
Markovian Score Climbing: Variational Inference with KL(p||q)
Christian A. Naesseth, Fredrik Lindsten, and David Blei
Advances in Neural Information Processing Systems 2020
```

The implementation uses Autograd https://github.com/HIPS/autograd. Each method can be run by downloading the relevant dataset and executing the corresponding Python script.

## Skew Normal
Toy model with no data.

## Bayesian Logistic Regression
There are three different datasets considered, Cleveland Heart Disease (https://archive.ics.uci.edu/ml/datasets/heart+disease), Ionosphere (https://archive.ics.uci.edu/ml/datasets/ionosphere), and Pima Indians Diabetes.

## Stochastic Volatility
Monthly returns over 10 years (9/2007 to 8/2017) for the exchange rate of 18 currencies with respect to the US dollar (https://www.federalreserve.gov/releases/h10/).