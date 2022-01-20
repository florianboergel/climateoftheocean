#!/usr/bin/env python
# coding: utf-8

# # Exercise 3, analyzing the Baltic Sea

# In[29]:


import matplotlib.pyplot as plt
import xarray as xr


# In[2]:


ds = xr.open_dataset("data/ocean_day3d.nc")


# When you start your work at IOW you will start by reading literature about the dynamics of the Baltic Sea. Soon you will notice that nearly every article starts with a paragraph similar to:
# 
# > "The hydrography of the Baltic Sea depends on the water exchange with the world ocean which is restricted by the narrows and sills of the Danish Straits and on river runoff into the Baltic [Meier and Kauker, 2003]."
# 
# 

# Look at this figure([Meier and Kauker, 2003](https://agupubs.onlinelibrary.wiley.com/cms/asset/144f12b6-8897-474b-b4e7-4dc1ad3c8faa/)) and zoom on the connection between the world ocean and the Baltic Sea.
# 
# The coordinates are : `lon=8-17, lat=53-59`
# 
# ![](https://agupubs.onlinelibrary.wiley.com/cms/asset/144f12b6-8897-474b-b4e7-4dc1ad3c8faa/jgrc9306-fig-0001.png)

# Check you data first. In MOM we use different names for lon, lat and depth:
# 
# lon = xt_ocean
# lat = yt_ocean
# depth = st_ocean
# 
# xarray allows you to select areas using
# 
# ```python
# 
# ds.sel(xt_ocean=slice(lon1, lon2), yt_ocean=slice(lat1, lat2))
# ```

# In[74]:


ds.dims


# In[72]:


danish_straits = ds.sel(xt_ocean=slice(,), yt_ocean=slice(,))


# In[12]:


# Instead of selecting coordinate, we can also use 
# index selction using .isel
# we are selecting the surface and the first timestep of 
# the variable salt and plot it

danish_straits.salt.isel(st_ocean=0, time = 0).plot()


# **Question 1:** What is the first thing you notice, when you compare to the realistic bathymetry above?
# 
# Answer: 

# **Question 2:** Look at the colobar. What role play the Danish Straits for the salinity of the Baltic Sea?
# 
# Answer:

# Following on common Baltic Sea introductions, you will find something similar to:
# 
# > The inflow of freshwater by river runoff and a positive net precipitation cause a positive water balance with respect to the North Sea. The positive water balance leads to strong gradients in salinity and ecosystem variables (Reckermann et al., 2008).

# So let's look at the mean surface salinity!

# In[91]:


level = [0,2,4,6,8,10,15,20,30,35]

f, ax = plt.subplots(1)

ds.salt.isel(st_ocean=0).mean("time").plot(levels=level, cmap=plt.cm.jet)


# Following Markus second sentence of his paper,
# 
# > In the long-term mean, high-saline water from the Kattegat enters the Baltic proper and low-saline water leaves the Baltic because of the freshwater surplus.
# 
# Think about the exchange flow of the Baltic Sea. How would the profile of a transect a 16°E look like? 

# In[129]:


# We first select the latitude range 53-57, then longitude at 16°E
# note that by using `method="nearest"` we will search for the nearest lon coordinate to 13

transect = ds.sel(yt_ocean=slice(53, 57)).sel(xt_ocean=16, method="nearest")


# This leaves us with the dimensions: time, depth and latitude

# In[95]:


transect.salt.dims


# Markus talks about the long-term mean in his paper. So start by averaging over the time dimension.
# 
# using 
# 
# ```python
# .mean("time")
# ```

# In[130]:


transect_mean_time = transect.mean("time")


# Now we can average over the latitude, to give us a depth profile.

# In[131]:


transect_mean_time_latitude = transect_mean_time.mean("yt_ocean")


# In[132]:


f, ax = plt.subplots(1)
transect_mean_time_latitude.salt.plot(ax=ax, y="st_ocean")
ax.set_ylabel("Depth [m]")
ax.set_xlabel("Salinity [g/kg]")
ax.invert_yaxis()


# Following along with Markus Paper:
# 
# > The bottom water in the deep subbasins is ventilated mainly by large perturbations, so-called major Baltic saltwater inflows [Matthäus and Franck, 1992; Fischer and Matthäus, 1996]. 

# In this year we have no strong inflow. However, we can notice inflows of high saline water analyzing the station located in the Arkona Basin.

# In[100]:


by2 = ds.sel(xt_ocean = 16.2, yt_ocean = 55.5, method="nearest")


# In[111]:


g = (by2.salt - by2.salt.mean("time")).plot(col="time", col_wrap = 3, y="st_ocean")

for ax in g.axes[0]:
    ax.invert_yaxis()
    ax.set_xlabel("Salinity[g/kg]")
    ax.set_ylabel("Depth [m]")

g.fig.tight_layout()


# **Question:** Focus on the month of February. What is happening?

# These saline inflows are important for the oxygen supply of the deeper layers, since due to the strong stratification in the Baltic Sea only layers above the permanent halocline are directly influenced by the atmosphere and therefore supplied with oxygen (Mohrholz et al., 2015).

# # Seasonal cycle of the temperature

# We will now look at the depth-averaged seasonal cycle of the Baltic Sea.

# In[136]:


ds_temp_season = ds.temp.resample(time="1M").mean("time").mean("st_ocean")


# In[137]:


(ds_temp_season-ds_temp_season.mean()).plot(col="time", col_wrap =3)


# **Question: ** Above you see the deviation from the mean temperature of the Baltic Sea for every single month. Try to discuss the differences for every month.

# ## Sea Ice
# 
# During winter time parts of the Baltic Sea are covered with sea ice. The sea ice influences the air-sea interaction. Sea ice directly influences temperature, salinity, but also the transfer of momentum into the ocean. The annual ice cover varies from only being present in the Bothnian Bay to a nearly fully covered Baltic Sea. Therefore, the Baltic Sea is exposed to great variation in sea ice cover.

# In[113]:


ice = xr.open_dataset("data/ice_day.nc")


# Let's start by analyzing the seasonal sea ice cover.

# In[117]:


ice


# In[128]:


ice.MI.resample(time="1M").mean().plot(col="time", col_wrap=4)


# **Question:** Why do we find sea ice in the North but not in the central Baltic Sea?
