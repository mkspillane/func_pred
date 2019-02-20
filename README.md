# func_pred

The general goal is to input time series data and get out predictions for future data.  The first steps are based on the idea
that if a model can predict an underlying ODE it can then accurately predict forward. This was started because out of the box
NN/LSTM didnt seem to be doing a very good job on simple functions like sin and exp(-x^2).



-nonlinear_coeff.py
Input function values for a domain of x the model will predict outside of that range.  

This is only expected to work well on functions that are solutions to ODE's with coeffients which are at most linear in x.  
Coeffients that are higher order polynomials in x could be ad hoc by simple changes of adding more inputs to forward_X.  
However, hopefully in future iterations a NN can be used so that nonlinear functions of x can work.

This particular model is equivilant to linear models with appropriate interaction terms, but that isnt along the lines of future planned steps and is also rather ad hoc.

Not all functions are obviously solutions to such ODEs, however, physics has been very sucessful using fairly simple systems
of PDEs 

-NN t and l.py
This is a NN with cells made up of a tanh activation with a linear skip:

  A_{i+1} = tanh(W_{i+1,t} A_{i}+b_{i+1,t})+W_{i+1,l}A_i+b_{i+1,l}
  
The final output is is then matched with the Y's as done above to produce a delta_Y output.  Two cost functions are provided one of which is commented out (L2).  Adding L1 would be fairly easy and adding multiple together would be easy because derivatives add.
  
This can have any number of layers depending on the input given.  There is a function which allows for the initialization of the NN with the number of layers and their dimensions to be chosen.  

The linear skip was added because a NN with tanh activations was having trouble for large extrapolations in testing on exp(-x^2).  It is possible that this was due to problems replicating the identity function using tanh (recall that exp(-x^2) is a solution to y' +2x y = 0.  Whatever the reason this version with the skip works better.  

Finally, this file contains a couple optimizers that work on the NN. (There may be an error in the Nosterov momentum) as it can give somewhat eratic responses.
