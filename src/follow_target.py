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

  def calc_circ_rect_centers(image, range_names):
    # The box could be entirely hidden by the robot, so we can't assume both orange regions are
    # visible and use watershedding. We would want to erode to remove noise, but the box is quite
    # small and eroding can erode it entirely

    def area_proportion(stats):
      return float(stats[cv2.CC_STAT_AREA]) / (stats[cv2.CC_STAT_HEIGHT] * stats[cv2.CC_STAT_WIDTH])

    def centre(stats):
      return np.array([
        stats[cv2.CC_STAT_LEFT] + stats[cv2.CC_STAT_WIDTH] / 2,
        stats[cv2.CC_STAT_TOP] + stats[cv2.CC_STAT_HEIGHT] / 2,
      ])

    def calc_circ_rect_center(colour_range):
      mask_threshold = cv2.inRange(image, *colour_range)
      mask_threshold_dilate = cv2.dilate(mask_threshold, dilate_kernel, iterations=3)
      _, _, stats, _ = cv2.connectedComponentsWithStats(mask_threshold_dilate)

      # Rough way to distinguish circle from rectangle if both quite visible: the rectangle will
      # have a larger proportion of pixels in its bounding box
      stats = stats[stats[:,cv2.CC_STAT_HEIGHT] != image.shape[0]]             # Remove background
      stats = stats[stats[:,cv2.CC_STAT_AREA].argsort()][-2:]                  # Take the largest two by area, in case of extra noise
      stats = stats[np.apply_along_axis(area_proportion, 1, stats).argsort()]  # Sort by proportion of area

      return \
        (centre(stats[0]), centre(stats[1])) if stats.shape[0] == 2 else \
        (None, None)

    return {
      range_name: calc_circ_rect_center(colour_ranges[range_name])
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

    orange_circ_center_1, orange_rect_center_1 = calc_circ_rect_centers(image_1, ('orange',))['orange']
    orange_circ_center_2, orange_rect_center_2 = calc_circ_rect_centers(image_2, ('orange',))['orange']

    logger.info('orange_circ_center_1: %s, orange_rect_center_1: %s', orange_circ_center_1, orange_rect_center_1)
    logger.info('orange_circ_center_2: %s, orange_rect_center_2: %s', orange_circ_center_2, orange_rect_center_2)

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
