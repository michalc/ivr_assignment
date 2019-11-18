ivr_assignment
==============

Separate nodes have been written for each part of the assignment: only one node at a time should be run.

```
rosrun ivr_assignment robot_vision_target_detection.py
```

The below files constrain link 3 to 0, and use geometry to determine the link angles from the camera images.

```
rosrun ivr_assignment robot_vision_joint_state.py
rosrun ivr_assignment robot_control_forward_kinematics.py
rosrun ivr_assignment robot_control_closed_loop.py
```

The below files do not have such a contstraint, and use a least-squares method to determine the four link angles.

```
rosrun ivr_assignment robot_vision_joint_state_least_squares.py
rosrun ivr_assignment robot_control_forward_kinematics_squares.py
rosrun ivr_assignment robot_control_closed_loop_squares.py
```
