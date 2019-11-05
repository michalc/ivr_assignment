#!/usr/bin/env python

import logging
import sys

import cv2
import message_filters
import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from robot_vision import calc_positions_and_angles

def main():
  rospy.init_node('robot_control', anonymous=True)
  bridge = CvBridge()

  joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
  joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
  joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
  joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  logger.addHandler(handler)

  def calc_k(q):
    # cos and sin functions take 1-indexed as in mathematical notation
    def c(i):
      return np.cos(q[i - 1])

    def s(i):
      return np.sin(q[i - 1])

    return np.array([
      +2*c(1)*s(3)*c(4) +3*c(1)*s(3) +2*s(1)*c(2)*s(4),
      +2*s(1)*s(3)*c(4) +2*s(1)*s(3) -2*c(1)*c(2)*s(4),
      -2*s(2)*s(4) +2*c(2)*c(3)*c(4) +3*c(2)*c(3) +2,
    ])

  def calc_jacobian(q):
    # cos and sin functions take 1-indexed as in mathematical notation
    def c(i):
      return np.cos(q[i - 1])

    def s(i):
      return np.sin(q[i - 1])

    return np.array([
      [
        -2*s(1)*c(3)*c(4) -3*s(1)*s(3) +2*c(1)*c(2)*s(4),
        -2*s(1)*s(2)*s(4),
        +2*c(1)*c(3)*c(4) +3*c(1)*c(3),
        -2*c(1)*s(3)*s(4) +2*s(1)*c(2)*c(4),
      ],
      [
        +2*c(1)*s(3)*c(4) +3*c(1)*s(3) +2*s(1)*c(2)*s(4),
        +2*c(1)*s(2)*s(4),
        +2*s(1)*c(3)*c(4) +3*s(1)*c(3),
        -2*s(1)*s(3)*s(4) -2*c(1)*c(2)*c(4),
      ],
      [
        0.0,
        -2*c(2)*s(4) -2*s(2)*c(3)*c(4) -3*c(2)*c(3),
        -2*c(2)*s(3)*c(4) +3*c(2)*s(3),
        -2*s(2)*c(4) -2*c(2)*c(3)*s(4),
      ],
    ])

  def constrain_link_3(q, jacobian):
    return np.delete(q, 2,), np.delete(jacobian, 2, 1)

  state = {
    't_-1': 0,
    'x_-1': np.array([0.0, 0.0, 0.0]),
  }

  def camera_callback(data_1, data_2):
    image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
    image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')

    positions_and_angles = calc_positions_and_angles(image_1, image_2)
    now = rospy.get_time()

    # We weren't able to determine the position.
    if positions_and_angles['orange_circ_center'] is None:
      return

    first_time = state['t_-1'] == 0
    x_t = positions_and_angles['orange_circ_center']
    dt = now - state['t_-1']
    dx = (x_t - state['x_-1']) / dt
    q = positions_and_angles['q']
  
    state['t_-1'] = now
    state['x_-1'] = x_t

    if first_time:
      return

    jacobian = calc_jacobian(positions_and_angles['q'])
    q_const, jacobian_cons = constrain_link_3(q, jacobian)
    jacobian_inv = np.linalg.pinv(jacobian_cons)

    q_d = q_const + dt * jacobian_inv.dot(dx) * 5
    logger.info('q_d %s', q_d)

    k = calc_k(positions_and_angles['q'])
    logger.info('k %s', k)

    joint1_pub.publish(Float64(data=q_d[0]))
    joint2_pub.publish(Float64(data=q_d[1]))
    joint3_pub.publish(Float64(data=0.0))
    joint4_pub.publish(Float64(data=q_d[2]))

  camera_1_sub = message_filters.Subscriber('/camera1/robot/image_raw', Image)
  camera_2_sub = message_filters.Subscriber('/camera2/robot/image_raw', Image)
  message_filters \
    .ApproximateTimeSynchronizer([camera_1_sub, camera_2_sub], queue_size=1, slop=0.01) \
    .registerCallback(camera_callback)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    logger.info('Shutting down')
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
