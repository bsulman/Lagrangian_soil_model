# Lagrangian_soil_model

Simulates soil organic matter decomposition by explicitly tracking tracer particles of organic matter and microbial biomass through a non spatially explicit set of soil pores representing different pore size classes. Multiple types of organic matter are represented, which can be transformed when sharing a pore with a microbe. All movement and transformation is calculated probabilistically.

Model simulations and plots of the results are contained in a single code file (individual_C_model.py). The code uses Python 3 and this version was developed using python version 3.7.6, numpy version 1.17.3, and matplotlib version 3.1.2.

Key parameters of the model include `transform_matrix` (which contains probabilities of transformation among different organic matter types), `move_probs_matrix` (which contains probabilities of moving into a pore size class for the different organic matter types), `prob_leaving` (the probability of any particle leaving a pore of a given size class), and `pore_distribution` (relative number of the different pore size classes).

To run the simulations and plot the output, just run the script with python:

      python individual_C_model.py
 
