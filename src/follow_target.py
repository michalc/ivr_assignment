#!/usr/bin/env python

import logging
import sys

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def main():
  rospy.init_node('follow_target', anonymous=True)
  bridge = CvBridge()

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  logger.addHandler(handler)

  def camera1_callback(_):
    logger.info('/camera1/robot/image_raw callback')

  def camera2_callback(_):
    logger.info('/camera2/robot/image_raw callback')

  rospy.Subscriber('/camera1/robot/image_raw', Image, camera1_callback)
  rospy.Subscriber('/camera2/robot/image_raw', Image, camera2_callback)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    logger.info('Shutting down')
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
