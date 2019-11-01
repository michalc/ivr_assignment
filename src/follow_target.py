#!/usr/bin/env python

import logging
import sys

import cv2
import message_filters
import numpy as np
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

  colour_ranges = {
    'yellow': ((0, 100, 100), (50, 255, 255)),
    'blue': ((100, 0, 0), (255, 50, 50)),
    'green': ((0, 100, 0), (50, 255, 50)),
    'red': ((0, 0, 100), (0, 0, 255)),
    'orange': ((75, 110, 130), (95, 175, 220)),
  }
  dilate_kernel = np.ones((5, 5), np.uint8)

  def threshold_centers(image, range_names):
    def threshold_center(colour_range):
      mask_threshold = cv2.inRange(image, *colour_range)
      # Dilation allows a better center of mass calculation, as long as we know there is only one
      # source of colour in the image, and the shape fairly regular
      mask_threshold_dilate = cv2.dilate(mask_threshold, dilate_kernel, iterations=3)
      M = cv2.moments(mask_threshold_dilate)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

    return {
      range_name: threshold_center(colour_ranges[range_name])
      for range_name in range_names
    }

  def camera_callback(data_1, data_2):
    # Allow exceptions to bubble up: they are logged automatically, and will
    # stop the rest of the callback running
    image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
    image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')

    centers_1 = threshold_centers(image_1, ('yellow', 'blue', 'green', 'red'))
    centers_2 = threshold_centers(image_2, ('yellow', 'blue', 'green', 'red'))

    logger.info('centers_1: %s', centers_1)
    logger.info('centers_2: %s', centers_2)

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
