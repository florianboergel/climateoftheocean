#!/usr/bin/env python
# coding: utf-8

# # Exercise 2, melting the snowball earth
# 
# > Start after finishing excercise 1 
# 
# - toc: False 
# - badges: true
# - comments: False
# - categories: [jupyter]

# In the last lectures we implemented a energy balance model. The energy balance equation is give by
# 
# \begin{gather}
# \color{brown}{C \frac{dT}{dt}}
# \; \color{black}{=} \; \color{orange}{\frac{(1 - α)S}{4}}
# \; \color{black}{-} \; \color{blue}{(A - BT)}
# \; \color{black}{+} \; \color{grey}{a \ln \left( \frac{[\text{CO}₂]}{[\text{CO}₂]_{\text{PI}}} \right)},
# \end{gather}
# 
# Recall that in the last lecture we implemented a simplied ice albedo feedback by allowing the albedo to depend on temperature:
# 
# $$\alpha(T) = \begin{cases}
# \alpha_{i} & \mbox{if }\;\; T \leq -10\text{°C} &\text{(completely frozen)}\\
# \alpha_{i} + (\alpha_{0}-\alpha_{i})\frac{T + 10}{20} & \mbox{if }\;\; -10\text{°C} \leq T \leq 10\text{°C} &\text{(partially frozen)}\\
# \alpha_{0} &\mbox{if }\;\; T \geq 10\text{°C} &\text{(no ice)}
# \end{cases}$$
# 

# One thing that we did not adress in the last lecture was the impact of CO$_2$ increase. We simply set:
# 
# $\ln \left( \frac{ [\text{CO}₂]_{\text{PI}} }{[\text{CO}₂]_{\text{PI}}} \right) = \ln(1) = 0$
# 
# We then evaluated how an increasing solar constant changes the equalibirum temperature on earth. In this excercise you keep $S$ at $1368 W/m^2$ and instead increase the $CO_2$ concentration.
# 
# Replot the bifurcations diagramm for $CO_2$ increase instead of solar radiation.
# 

# In[ ]:


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from IPython.display import HTML
from IPython.display import display

from energy_balance_model import ebm


# In[ ]:


def CO2_change(t): 
    return 280 + co2


# Note that `co2` is a global variable here. In general all variables that are assigned in a function call are private (only accessible within the function). However, if this variable is not defined within the function, python looks for a global variable that is called `co2` (in your complete code). Therefore, remember to increase `co2` for the following task.
# 
# Start by running the model for with different C02 concentrations and plot the equlibrium temperature. 
# 
# Hint: You can copy nearly all code from the Lecture *Snowball earth*

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


plt.figure(figsize = (8,6))
plt.plot(co2vec[0:len(co2vec)//2], tvec[0:len(co2vec)//2], color = "blue", label = "cool branch", alpha = 0.3)
plt.plot(co2vec[len(co2vec)//2:], tvec[len(co2vec)//2:], color = "red", label = "warm branch", alpha = 0.3)
plt.axvline(1368, color = "yellow", lw = 5, alpha = 0.2, label = "Pre-industiral / present insolation")

plt.plot(420, 14, marker="o", label="Our preindustrial climate", color="orange", markersize=8)
plt.plot(420, -38, marker="o", label="Alternate preindustrial climate", color="lightblue", markersize=8)
plt.plot(280, -48, marker="o", label="neoproterozoic (700 Mya)", color="lightgrey", markersize=8)

plt.xlabel("CO$_2$ concentration [ppm]")
plt.ylabel("Global temperature T [°C]")

plt.legend()
plt.grid()

