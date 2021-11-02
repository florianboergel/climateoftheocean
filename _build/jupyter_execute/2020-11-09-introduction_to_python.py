#!/usr/bin/env python
# coding: utf-8

# # Lecture 1, Introduction to python

# # Getting started
# 
# ## Introduction to Python
# 
# ![](python.png)
# 
# 
# ## What are we doing during the course?
# 
# The course aims to introduce you to the first steps of data analysis
# 
# - save data
# - organize and manipulate data
# - further tools for data analysis
# 
# If you never coded before than starting with Python is perfect! Python is one of the easiest and most straight forward languages.

# In[ ]:


x = 1


# In[ ]:





# In[ ]:


x = "abcd"


# In[ ]:





# x as an integer and then a string? The same concept is applicable for many other different examples. Python has a minimalistic approech and follows a simple syntax. This is why source code is very easy to read.

# In[ ]:


x = 0
if x > 0:
    statement = "x is positive"
elif x < 0:
    statement = "x is negative"
else: 
    statement = "x is zero or none"
    
print(statement)


# In[ ]:


x = 5 - 4       # Comments are made with a hash Raute
y = "Hello"     # Everything after the hash will be a comment
if y == "hallo":
    z = x * 2
    y = y + " World" # This is how you combine strings!

print("x :", x) # The letter x and the our variable x
print("y :", y)


# **First summary:**
# - Indenting of the source code has a meaning!
#     -  the indentation of your code organizes it into blocks within blocks within blocks. 
# - the first assignment of a variable creates it
#     - we don't care if it is an integer, float or string
# - assignments of variables use *=*, to compare two variables we use *==*
# - also: logical operators are words (and, or, not) *not* symbols

# ## Variables and types
# ### Variable
# The value of a variable can be obtained by writing its name.

# In[ ]:


pi = 3.14

radius = 68.7 


# In[ ]:





# **Example:**
# 
# Calculate the area of the circle 
# 
# A = $\pi r^2$ 
# 
# Exponentiation in Python is defined as
# 
# r**2 
# 
# 

# In[ ]:


area = 
print(area)


# ### Types
# 
# Without going into detail:

# In[ ]:


pi = 3.141516546859754674896794
days_of_week = 5
x = 'Hey Guys'
y = "Also works this way ..."
z = True

print('pi: ', type(pi))
print('days_of_week: ', type(days_of_week))
print('x: ', type(x))
print('y: ', type(y))
print('z: ', type(z))


# Does this really matter? Actually, no, since Python's use of variables is very intuitive. Still, keep in mind that a different variable type can lead to different behaviour:

# In[ ]:


2 + 3 


# In[ ]:


'ab' + 'cd'


# In[ ]:


# We define two integers and assign the division to c
a = 1
b = 5

c = a / b


# What is the value of c?

# In[ ]:


print(c)


# ### Lists
# As opposed to int, bool etc., a list is a compound data type; you can group values together:
# 
# 
# ```python
# a = "is"
# b = "nice"
# my_list = ["my", "list", a, b]
# ```
# After measuring the height of your family, you decide to collect some information on the house you're living in. The areas of the different parts of your house are stored in separate variables for now, as shown in the script.

# ### List of lists
# As a data scientist, you'll often be dealing with a lot of data, and it will make sense to group some of this data.
# 
# Instead of creating a flat list containing strings and floats, representing the names and areas of the rooms in your house, you can create a list of lists. The script on the right can already give you an idea.
# 
# Don't get confused here: "hallway" is a string, while hall is a variable that represents the float 11.25 you specified earlier.

# In[ ]:


# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom"], bed]


# ### Example
# 
# - Print out house
# - Print out the type of house
# - Add the the bath to the house
# 

# In[ ]:





# ## Subset and conquer
# Subsetting Python lists is a piece of cake. Take the code sample below, which creates a list x and then selects "b" from it. Remember that this is the second element, so it has index 1. You can also use negative indexing.
# 
# ```python
# x = ["a", "b", "c", "d"]
# x[1]
# x[-3] # same result!
# ```
# 
# ### Example
# Remember the areas list from before, containing both strings and floats? Its definition is already in the script. Can you add the correct code to do some Python subsetting?

# In[ ]:





# ### Tip
# 
# Some times it makes sense to avoid lists of lists and instead use dictionaries.
# 
# ```python
# house = {
#     "hallway": hall,
#     "kitchen": kit,
#     "living room": liv,
#     "bedroom": bath
# }
# ```

# In[ ]:


house = {
    "hallway": hall,
    "kitchen": kit,
    "living room": liv,
    "bedroom": bed
}


# In[ ]:


house.keys()


# In[ ]:


house["hallway"]


# ### Familiar functions
# Out of the box, Python offers a bunch of built-in functions to make your life as a data scientist easier. You already know two such functions: print() and type(). You've also used the functions str(), int(), bool() and float() to switch between data types. These are built-in functions as well.
# 
# Calling a function is easy. To get the type of 3.0 and store the output as a new variable, result, you can use the following:
# 
# result = type(3.0)
# 
# The general recipe for calling functions and saving the result to a variable is thus:
# 
# output = function_name(input)

# In[ ]:


# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = False

# Print out type of var1
print(type(var1))
# Print out length of var1
print(len(var1))

# Convert var2 to an integer: out2
int(var2)


# ## Numpy
# 
#  Numpy is a library, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# In[ ]:


import numpy as np

# List
a = [1,2,3,4,5,6,7,8,9]

# numpy array
A = np.array([1,2,3,4,5,6,7,8,9])

print("This is a list: {} and looks like \n {}".format(type(a), a))
print("This is an array: {} and looks like \n {}".format(type(A), A))


# ### Create arrays of a give length

# In[ ]:


y_ar = np.arange(0, 1, 0.1)
print("Lenght of array y_ar is {}.".format(len(y_ar)))
print(y_ar)

x_ar = np.linspace(1, 10, 5) #creates an array of length 5 between 1 and 10 

print("Lenght of array x_ar is {}.".format(len(x_ar)))
print(x_ar)


# ### Multidimensional arrays

# In[ ]:


z_ar = np.zeros((100)) #creates an array of shape (100,) with zeros

print("Shape of z_ar is {}".format(z_ar.shape)) 

z_ar = np.zeros((100,1))

print("Shape of z_ar is {}".format(z_ar.shape))

z_ar = np.zeros((100, 1, 3, 5))

print("Shape of z_ar is {}".format(z_ar.shape))

z_ar[0, 0, 1, 2]


# .shape gives you the dimensions of the array, while len() only returns the lenght of the first dimension!

# In[ ]:


import time

a = [1, 2, 3]
b = [3, 4, 5]

print(a + b) 

c = []

for count in range(len(a)):
    c.append(a[count] + b [count])

print(c)


# In[ ]:


a = np.array([1., 2., 3.])
b = np.array([3., 4., 5.])

print(a + b)


# ### Why do we do this?

# In[ ]:


a = [1 for x in range(1000000)]
b = [1 for x in range(1000000)]


time1 = time.time()
c = []
for count in range(len(a)):
    c.append(a[count] + b[count])
time2 = time.time()   
print('This took %0.3f ms' % ((time2-time1)*1000.0))


a = np.ones(1000000)
b = np.ones(1000000)

time1 = time.time()
c = a + b
time2 = time.time()   
print('This took %0.3f ms' % ((time2-time1)*1000.0))


# ### Useful functions:
# 

# In[ ]:


z = np.random.rand(10, 20, 30)

print(z.shape)


# In[ ]:


z_mean = np.nanmean(z, axis = 1)

print(z_mean.shape)

z_mean = np.nanmean(z, axis = (1,2))

print(z_mean.shape)

z_mean_sqrt = np.sqrt(z_mean) # square root


# ## masked arrays

# In[ ]:


z = np.random.rand(20, 20)
mask = (z < 0.5)

print(mask)
print(mask.shape)


# In[ ]:


z_masked = np.ma.asarray(z)
z_masked.mask = mask

print(z_masked)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.pcolor(z)


# In[ ]:


plt.pcolor(z_masked)


# ## Functions

# In[ ]:


def multiply_constant(x, constant = 5):
    return x * constant

print(multiply_constant(5))
print(multiply_constant(5, 3))
print(multiply_constant(x = 5, constant = 3))
print(multiply_constant(a, constant = 2.5))


# # Data visualization

# ## 1D Plot

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(0,300) # Numbers from 0 to 30 (30 not included)
y = np.sin(np.arange(0,300)) + np.random.normal(0,5,300) # Create sinus
                                                       ##signal and add noise

dataFrame = pd.DataFrame({'Intervals': x,
                         'values': y})    

rolling_mean = dataFrame['values'].rolling(10, min_periods=1, center=True).mean()

plt.plot(x,y,label='Kurve', alpha=0.5)
plt.plot(x, rolling_mean.values, label = 'Rolling Mean')
plt.title('Sinus + Random')
plt.xlabel('Interval')
plt.ylabel('Values')
plt.legend()


# ## 2D Plots

# In[ ]:


z = np.random.rand(20,20)

levels = np.linspace(0, 1, 10)
f, ax = plt.subplots(1)
im = ax.pcolor(z)
#im = ax.contourf(z, levels = levels)
#ax.contour(z)
ax.set_title('Random Noise from 0 - 1')
f.colorbar(im)


# # netCDF4-files

# In[ ]:


import xarray as xr

url = "https://ds.nccs.nasa.gov/thredds/dodsC/CMIP5/ESGF/GISS/rcp45/E2-R_rcp45_r6i1p3_day/tos_day_GISS-E2-R_rcp45_r6i1p3_20910101-21001231.nc"

ds = xr.open_dataset(url)


# In[ ]:


ds.isel(time = 1)


# In[ ]:


ds.isel(time = 1).tos.plot()


# In[ ]:


tmp = ds.sel(lon = slice(150,200)).isel(time = slice(1,100)).tos.mean(["lon","lat"])


# In[ ]:


tmp.plot()


# In[ ]:


tmp.rolling(time = 10, min_periods =1).mean().plot()

