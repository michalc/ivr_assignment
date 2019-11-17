#!/usr/bin/env python

import traceback

import message_filters
import numpy as np
import rospy
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from shared import calc_k, calc_jacobian, calc_positions_and_angles

K_p = np.array([
  [2.0, 0.0, 0.0],
  [0.0, 2.0, 0.0],
  [0.0, 0.0, 2.0],
])
K_d = np.array([
  [0.1, 0.0, 0.0],
  [0.0, 0.1, 0.0],
  [0.0, 0.0, 0.1],
])

def main():
  rospy.init_node('robot_control', anonymous=True)
  bridge = CvBridge()

  joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
  joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
  joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
  joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
  k_pub = rospy.Publisher("/k", Float64MultiArray, queue_size=10)

  def constrain_link_3(q, jacobian):
    return np.delete(q, 2,), np.delete(jacobian, 2, 1)

  state = {
    't_-1': 0,
    'x_-1': np.array([0.0, 0.0, 0.0]),
    'e_-1': np.array([0.0, 0.0, 0.0]),
  }

  def camera_callback(data_1, data_2):
    try:
      image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
      image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')

      positions_and_angles = calc_positions_and_angles(image_1, image_2)
      now = rospy.get_time()

      # We weren't able to determine the position.
      if positions_and_angles['target_center'] is None:
        return

      first_time = state['t_-1'] == 0
      x_t = positions_and_angles['target_center']
      x_e = positions_and_angles['joint_centers']['red']
      e_t = x_t - x_e

      dt = now - state['t_-1']
      if dt == 0:
        return
      dx = (x_t - state['x_-1']) / dt
      de = (e_t - state['e_-1']) / dt

      state['t_-1'] = now
      state['x_-1'] = x_t
      state['e_-1'] = e_t

      if first_time:
        return

      q = positions_and_angles['q']
      jacobian = calc_jacobian(positions_and_angles['q'])
      q_const, jacobian_cons = constrain_link_3(q, jacobian)
      jacobian_inv = np.linalg.pinv(jacobian_cons)

      q_d = q_const + dt * jacobian_inv.dot(K_p.dot(e_t) + K_d.dot(de))
      print('q_d', q_d)

      # The inverse kinematics doesn't know about contraints/allowed robot
      # Configurations
      if q_d[0] > np.pi:
        q_d[0] -= 2 * np.pi
      if q_d[0] < -np.pi:
        q_d[0] += 2 * np.pi
      q_d[1] = max(q_d[1], 0.0)

      joint1_pub.publish(Float64(data=q_d[0]))
      joint2_pub.publish(Float64(data=q_d[1]))
      joint3_pub.publish(Float64(data=0.0))
      joint4_pub.publish(Float64(data=q_d[2]))

      k = calc_k(np.array([q_d[0], q_d[1], 0.0, q_d[2]]))
      k_pub.publish(Float64MultiArray(data=k))
    except Exception:
      traceback.print_exc()

  camera_1_sub = message_filters.Subscriber('/camera1/robot/image_raw', Image)
  camera_2_sub = message_filters.Subscriber('/camera2/robot/image_raw', Image)
  message_filters \
    .ApproximateTimeSynchronizer([camera_1_sub, camera_2_sub], queue_size=1, slop=0.01) \
    .registerCallback(camera_callback)

  rospy.spin()

if __name__ == '__main__':
    main()
