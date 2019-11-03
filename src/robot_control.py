#!/usr/bin/env python

import logging
import sys

import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from robot_vision import calc_positions_and_angles

def main():
  rospy.init_node('robot_control', anonymous=True)
  bridge = CvBridge()

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
      +2*c(1)*s(3)*c(4) +2*c(1)*s(3) +2*s(1)*c(2)*s(4),
      +2*s(1)*s(3)*c(4) +2*s(1)*s(3) -2*c(1)*c(2)*s(4),
      -2*s(2)*c(3)*c(4) -3*s(2)*c(2)*c(4) +2,
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

  def camera_callback(data_1, data_2):
    image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
    image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')

    positions_and_angles = calc_positions_and_angles(image_1, image_2)

    k = calc_k(positions_and_angles['q'])
    jacobian = calc_jacobian(positions_and_angles['q'])

    logger.info('k %s', k)
    logger.info('jacobian %s', jacobian)

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
