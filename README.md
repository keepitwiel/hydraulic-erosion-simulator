# Hydraulic Erosion Simulator

A python implementation of the paper [Fast Hydraulic Erosion Simulation and Visualization on
GPU](https://hal.inria.fr/inria-00402079/document).

To run, install the appropriate packages and run one of the examples provided.

You will see output similar to this:
![This was generated using example 4](screenshot.jpg)

## Engine & Algorithm
The algorithm (src/fast_erosion_algorithm.py) was implemented from the above paper with a few additions
found in the authors' implementation, which can be found 
[here](https://github.com/Huw-man/Interactive-Erosion-Simulator-on-GPU).

The engine (src/fast_erosion_engine.py) is nothing more than a container that holds all the fields required
to run the algorithm.
