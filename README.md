# func_pred

The general goal is to input time series data and get out predictions for future data.  The first steps are based on the idea
that if a model can predict an underlying ODE it can then accurately predict forward. This was started because out of the box
NN/LSTM didnt seem to be doing a very good job on simple functions like sin and exp(-x^2).



nonlinear_coeff.py
Input function values for a domain of x the model will predict outside of that range.  

This is only expected to work well on functions that are solutions to ODE's with coeffients which are at most linear in x.  
Coeffients that are higher order polynomials in x could be ad hoc by simple changes of adding more inputs to forward_X.  
However, hopefully in future iterations a NN can be used so that nonlinear functions of x can work.

This particular model is equivilant to linear models with appropriate interaction terms, but that isnt along the lines of future 
planned steps and is also rather ad hoc.

Not all functions are obviously solutions to such ODEs, however, physics has been very sucessful using fairly simple systems
of PDEs 
