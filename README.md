This project contains a Jupyter notebook implementing and visualizing the Brusselator and Gray–Scott reaction–diffusion models, with options to generate animations as GIFs.
​

## 1. Requirements
Create a new Python environment and install:
1) Python 3.9+
2) NumPy
3) SciPy
4) Matplotlib
5) imageio
​

## 2. Running the code
You can run the notebook from top to bottom using “Run All” or by executing each cell in order.​

In any case you have to run the "Modules" cell first. 

​
The Brusselator and Gray-scott parts can be run independently. 
1) The cell "Creation of a class for the model", well... create a class for the model
2) The cell "Runs the model and plot the result" is self explanatory as well, with the following arguments :
   * Size : the size of the grid on which the model will be ploted
   * T : total simulated time in arbitrary unit
3) The cell "Create and save a GIF of the model" has the following arguments :
   * filename : the name you want the gif to have. Do not forget the ".gif" at the end
   * every :  Interval in time steps between frames kept for the GIF
   * duration : Display time in seconds of each frame in the GIF animation
  
   &rarr; The GIF file will be written to the current working directory and can be opened with any image viewer that supports GIFs
