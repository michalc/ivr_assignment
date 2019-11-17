#!/usr/bin/env python

import traceback

import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge

from shared import calc_positions_and_angles


def main():
  rospy.init_node('robot_vision_joint_state', anonymous=True)
  target_pub = rospy.Publisher("target_estimate", Float64MultiArray, queue_size=10)
  bridge = CvBridge()

  def camera_callback(data_1, data_2):
    try:
      image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
      image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')
      target_center = calc_positions_and_angles(image_1, image_2)['target_center']
      print('target_center', target_center)
      target_pub.publish(Float64MultiArray(data=target_center))
    except Exception as ex:
      traceback.print_exc()

  camera_1_sub = message_filters.Subscriber('/camera1/robot/image_raw', Image)
  camera_2_sub = message_filters.Subscriber('/camera2/robot/image_raw', Image)
  message_filters \
    .ApproximateTimeSynchronizer([camera_1_sub, camera_2_sub], queue_size=1, slop=0.01) \
    .registerCallback(camera_callback)

  rospy.spin()

if __name__ == '__main__':
    main()
