# Guidance and Control for Polaris GEM car at UIUC Highbay

This repository provides guidance for autonomous driving development on the Polaris GEM series vehicles at the UIUC Highbay. It includes a brief introduction to the GEM car and an overview of the implemented control framework. The system is tailored for the GEM e4 and can be adapted to the e2 with minor modifications.

# Gem Car Info

Home Page: https://publish.illinois.edu/robotics-autonomy-resources/

We mainly use the GEM e2 and e4 vehicles at Highbay. For details, contact John at jmhart3@illinois.edu. Highbay provides a pre-configured Ubuntu 20.04 system for direct vehicle use, and you can ask the administrator to help copy the SSD. We also have an ACRL SSD with our custom setup that is ready to use. It is stored together with the car keys, and John can help you locate it. Please note that some vehicle hardware has changed, so make sure to follow the latest updates.

# Requirements for Real Vehicle Testing

You must reserve a time slot using the following link: https://robotics.illinois.edu/reserve-lab/

After logging into your Illinois account, you can request training through the Highbay reservation system or contact John directly to schedule it. Training is required before participating in vehicle testing. Each test must involve at least two people: one driver and one observer. Both roles require completing the training, which covers safety procedures, observer duties, and basic driving. To serve as a driver, you must also pass a short driving test. Lab safety certifications must be up to date on the university’s website.

Once you complete the training, you are allowed to freely reserve time slots for testing. However, please note that priority is given to lab sessions for ECE 484 and CS 588.

# Simulation

Hang and Jiaming have developed a simulation environment for the GEM e2 vehicle, available at https://github.com/hangcui1201/POLARIS_GEM_e2_Simulator.

We have extended the simulator with several control methods. In the /simulation folder, both DDP and Acados-based MPC implementations are provided and can run in this environment. Although the GEM e2 and e4 differ in hardware, they can be treated as equivalent in simulation. Note that the simulator uses speed and front wheel angle as control inputs, which differ from the real vehicle. Despite this, the simulation can still reflect system behavior to a useful extent.

The system includes plotting functions for visualizing trajectories and control outputs. Obstacles can be added by specifying points in the simulation world, which allows for testing basic avoidance scenarios.

# MPC interface

The e4 interface can be used directly on both the GEM e4 and e2 vehicles. We have finished a ROS-based version, which is already included on our SSD. You can use it as the main src folder to run the system. This interface sends throttle, brake, and steering commands through the PACMod system. If you are using our SSD, the setup is ready to use. We have also implemented an MPC framework based on occupancy grids. The obstacle data is generated from simulated LiDAR measurements in the simulation environment. You can adjust the behavior or extend the implementation based on the provided code. 

In the ROS version, you can also find different MPC implementations written in Python. They can be run directly. Please note that the Acados library must be installed separately by following the official Acados documentation.
