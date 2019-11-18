#!/usr/bin/env python

import traceback

import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge

from shared import calc_positions_and_angles_least_squares


def main():
  rospy.init_node('robot_vision_join_state', anonymous=True)
  joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
  joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
  joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
  joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
  q_pub = rospy.Publisher("q", Float64MultiArray, queue_size=10)
  bridge = CvBridge()

  state = {
    'q': None,
    'desired_joint_config': [
      # As in lecture can be between 0 and 1
      np.array([0.0, 0.0, 0.0, 0.0]),
      np.array([0.2, 0.0, 0.0, 0.0]),
      np.array([0.2, 0.2, 0.0, 0.2]),
      np.array([0.4, 0.2, 0.0, 0.2]),
      np.array([0.4, 0.4, 0.0, 0.2]),
      np.array([0.4, 0.4, 0.0, 0.4]),
      np.array([0.6, 0.4, 0.0, 0.4]),
      np.array([0.6, 0.6, 0.0, 0.4]),
      np.array([0.8, 0.8, 0.0, 0.8]),
      np.array([1.2, 1.2, 0.0, 1.2]),
    ],
  }

  def camera_callback(data_1, data_2):
    try:
      image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
      image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')
      state['q'] = calc_positions_and_angles_least_squares(image_1, image_2)['q']
    except Exception as ex:
      traceback.print_exc()

  camera_1_sub = message_filters.Subscriber('/camera1/robot/image_raw', Image)
  camera_2_sub = message_filters.Subscriber('/camera2/robot/image_raw', Image)
  message_filters \
    .ApproximateTimeSynchronizer([camera_1_sub, camera_2_sub], queue_size=1, slop=0.01) \
    .registerCallback(camera_callback)

  def move_robot(_):
    try:
      joint_config = state['desired_joint_config'].pop(0)
    except IndexError:
      rospy.core.signal_shutdown('No more joint configurations')
      return

    print('Moving to:', joint_config)
    joint1_pub.publish(Float64(data=joint_config[0]))
    joint2_pub.publish(Float64(data=joint_config[1]))
    joint3_pub.publish(Float64(data=joint_config[2]))
    joint4_pub.publish(Float64(data=joint_config[3]))
    rospy.sleep(3)
    print('Estimated angles:', state['q'])
    q_pub.publish(Float64MultiArray(data=state['q']))
    rospy.Timer(rospy.Duration(1), move_robot, oneshot=True)

  rospy.Timer(rospy.Duration(3), move_robot, oneshot=True)
  rospy.spin()

if __name__ == '__main__':
    main()
