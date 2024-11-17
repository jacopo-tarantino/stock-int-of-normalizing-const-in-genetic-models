# Stochastic approximation of normalizing constants in genetic models with selection
The analysis of population genetics and the evolution of allele frequencies over time can be
framed as a filtering problem within the Hidden Markov Model (HMM) framework. To find a
computable filter, we follow the work of Papaspiliopoulos and Ruggiero (2014), who established
duality as a sufficient condition for filtering. We focus on the K-allele model and identify its dual,
building on the work of Barbour et al. (2000). Through this duality, the core problem reduces
to simulating a birth-and-death process and calculating its transition rates. When selection is
introduced into the model, the tractability of these rates diminishes, as they depend on the ratio
of multivariate density functions. These densities are the product of a normal distribution and a
Dirichlet distribution, both defined over an n-dimensional simplex. We propose various methods
to compute these ratios and compare their performances. First, we compute the normalizing
constants for the numerator and denominator separately using Monte Carlo integration with
importance sampling. We compare our approximations and their computational costs to the
analytical method of nested integration proposed by Genz and Joyce (2000). To address the bias
introduced by the first approach, we turn to direct approximations of the ratio using Annealed
Importance Sampling (AIS) and Linked Importance Sampling (LIS), as described by Neal (2005).
Finally, we evaluate all methods based on accuracy and computational time, ultimately defining
the optimal approach for the K-allele model.

The work has been supervised by Professor Matteo Ruggiero

## Files
> `project.py`: contains the full  thesis pdf.
> 
> `methods.py`: contains the AIS, LIS methodologies, the necessary markov chains (MH) and the respective computations corrections
>
> `simulations.py`: contains the MC integrations with Importance sampling
>
> `stoch_int_to_run.ipynb`: runs AIS and LIS, evaluating and comparing their accucary and computational cost
>
> `nested_int_to_run.ipynb`: defines and runs Nested Analytical Integration 


