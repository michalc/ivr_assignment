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
    # Ranges handle the slight shadow towards the bottom of the objects
    'yellow': ((0, 100, 100), (50, 255, 255)),
    'blue': ((100, 0, 0), (255, 50, 50)),
    'green': ((0, 100, 0), (50, 255, 50)),
    'red': ((0, 0, 100), (50, 50, 255)),
    'orange': ((75, 110, 130), (95, 175, 220)),
  }
  dilate_kernel = np.ones((5, 5), np.uint8)

  def calc_threshold_centers(image, range_names):
    def calc_threshold_center(colour_range):
      mask_threshold = cv2.inRange(image, *colour_range)
      # Dilation gives a better center of mass calculation, since it partially compensates for
      # occlusion, as long there is only one source of the colour, and the real-world shape is
      # fairly regular
      mask_threshold_dilate = cv2.dilate(mask_threshold, dilate_kernel, iterations=3)
      M = cv2.moments(mask_threshold_dilate)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

    return {
      range_name: calc_threshold_center(colour_ranges[range_name])
      for range_name in range_names
    }

  def calc_connected_comp_bounds(image, range_names):
    def calc_connected_comp_bound(colour_range):
      mask_threshold = cv2.inRange(image, *colour_range)
      mask_threshold_dilate = cv2.dilate(mask_threshold, dilate_kernel, iterations=3)
      _, _, stats, _ = cv2.connectedComponentsWithStats(mask_threshold_dilate)
      # From assumption of the environment, one component will have the same bounds as the image
      return stats[stats[:,cv2.CC_STAT_HEIGHT] != image.shape[0]]

    return {
      range_name: calc_connected_comp_bound(colour_ranges[range_name])
      for range_name in range_names
    }

  def camera_callback(data_1, data_2):
    # Allow exceptions to bubble up: they are logged automatically, and will
    # stop the rest of the callback running
    image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
    image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')

    threshold_centers_1 = calc_threshold_centers(image_1, ('yellow', 'blue', 'green', 'red'))
    threshold_centers_2 = calc_threshold_centers(image_2, ('yellow', 'blue', 'green', 'red'))

    logger.info('threshold_centers_1: %s', threshold_centers_1)
    logger.info('threshold_centers_2: %s', threshold_centers_2)

    # The box could be entirely hidden by the robot, so we can't assume both orange regions are
    # visible and use watershedding. We would want to erode to remove noise, but the box is quite
    # small and eroding can erode it entirely
    connected_comp_bounds_1 = calc_connected_comp_bounds(image_1, ('orange',))
    connected_comp_bounds_2 = calc_connected_comp_bounds(image_2, ('orange',))

    logger.info('connected_comp_bounds_1: %s', connected_comp_bounds_1)
    logger.info('connected_comp_bounds_2: %s', connected_comp_bounds_2)

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
