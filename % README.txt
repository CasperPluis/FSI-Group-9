% README
% Velocity measurements of open-channel flow
% bas Hofland, 30-10-2026
 
The files in this directory describe the measured velocities (2 components, streamwise-vertical) in an open channel flow in the Hydraulic Engineering lab at TU Delft. They are used in the assignment for the unit Fluid-Structure-Interaction CIEM4000-26.

Each of the files contains a measurement at one point in a vertical profile of a steady current over a rough bed. The names of the files indicate the dimensionless height from the bed at which the velocities were measured, as percentage of the water depth. For instance the data in file 'Z_23.3.dat' is measured at an elevation of z = 0.233*h from the bed.

The ASCII files contain three columns with dimensionless data of, respectively:
column 1: time, t~
column 3: streamwise velocity, u~
column 3: upward velocity, v~

The values were normalized by the mean velocity U and water depth h. When these values are obtained from the assignment, you can obtain the values with dimensions as follows:
t = t~ * h/U
u = u~ * U
v = v~ * U
