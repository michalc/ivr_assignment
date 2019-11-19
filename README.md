ivr_assignment
==============

## Requirements

rospy, numpy, scipy and opencv are required.


## Running

Separate nodes have been written for each part of the assignment: only one node at a time should be run.

```
rosrun ivr_assignment robot_vision_target_detection.py
```

The below nodes constrain link 3 to 0 degrees, and use geometry to determine the link angles from the camera images.

```
rosrun ivr_assignment robot_vision_joint_state.py
rosrun ivr_assignment robot_control_forward_kinematics.py
rosrun ivr_assignment robot_control_closed_loop.py
```

The below nodes do not have such a constraint, and use a least-squares method to determine the four link angles.

```
rosrun ivr_assignment robot_vision_joint_state_least_squares.py
rosrun ivr_assignment robot_control_forward_kinematics_least_squares.py
rosrun ivr_assignment robot_control_closed_loop_squares.py
```

During the run of `robot_control_closed_loop.py`, the concurrent node `error.py` can be run to output basic statistics on the error between the effector position and the target position.
