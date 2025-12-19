# 4-Wheeled FWD Ackermann Project #
This project is an example of a dynamic vehicle model for a 4-Wheeled Forward Wheel Drive Ackermann system done with a Jupyter Notebook Windows environment. Below will be a step-by-step tutorial for standard installations and setup along with a brief description of the classes and scripting cells.

## Setup ##
Download and install miniconda via: https://www.anaconda.com/docs/getting-started/miniconda/install

Once you have it installed, create an environment within the terminal and activate it:
- `conda create --name myenv`
- `conda activate myenv`

> [!NOTE]
> Compatible only with versions Python 3.9 or later.

Run these installs with your environment active:
`conda install numpy matplotlib=3.9.2 ipython`

## Class Descriptions ##
`robot_models.py`

Defines the movement of the car with two classes: `FourWheelKinematic` and `DynamicRobot`. The kinematic class assumes perfect grip and calculates the wheel geometry, electronic differential, and states. The dynamic class uses Euler-Lagrange equations for weight, inertia, and tire slide. A state vector is used to track angular velocity and velocity with a 4-wheeled visualizer.

`controllers.py`

A Lyapunov Stability-based controller that looks at the polar coordinates based on $$\rho$$, $$\alpha$$, and $$\beta$$ (corresponding to distance, angle, and orientation respectively). It calculates the Lyapunov Energy Function based on *V* which is the Total Error in the system. Reverse logic, tuning gains, and an Ackermann Conversion for steering angles are dictated as well.

`maps.py`

The `OccupancyGridMap` class creates a 2D area of cells that assign a cost value to how close you are to the blocked area (obstacles). A safety margin around the obstacles is applied so that the path finding algorithm will not cut too closely to the obstacles and account for vehicle geometry. An exponential decay gradient map is utilized to heavily punish the finding algorithm for being close to the obstacle while being a low cost for just barely being within the `safety_margin`. Obstacles can be spawned in using the `add` methods and there is a checker to see if obstacles coincide with starting/ending points.

`planner.py`

A Hybrid A* Planner and a Node system is utilized to find our paths and save Robot states for comparison. The `move` for physics simulation takes its current position and applies a steering angle over a short distance defined by `MOTION_STEP`. You can configure this for more/less steps for greater accuracy and cost of runtime. A cost function is utilized to sort all possible paths and pick the most efficient with steering, switching, and heading penalties. A standard A* search loop checks for queues, collisions, and thresholds.

More about results and scripting cells [here](https://docs.google.com/presentation/d/1c6GhVI54nvZKM4aBp5AJl54hNxxpbZu1547ZddBCIbQ/edit?usp=sharing) 
