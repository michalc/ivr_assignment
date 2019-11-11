#!/usr/bin/env python

import traceback

import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge

from robot_vision import calc_positions_and_angles


def calc_k(q):
  # cos and sin functions take 1-indexed as in mathematical notation
  def c(i):
    return np.cos(q[i - 1])

  def s(i):
    return np.sin(q[i - 1])

  return np.array([
    3*c(1)*s(3) + 2*c(2)*s(1)*s(4) + 3*c(3)*s(1)*s(2) + 2*c(4)*(c(1)*s(3) + c(3)*s(1)*s(2)),
    -2*c(1)*c(2)*s(4) - 3*c(1)*c(3)*s(2) + 2*c(4)*(-c(1)*c(3)*s(2) + s(1)*s(3)) + 3*s(1)*s(3),
    2*c(2)*c(3)*c(4) + 3*c(2)*c(3) - 2*s(2)*s(4) + 2,
  ])


def main():
  rospy.init_node('robot_vision_join_state', anonymous=True)
  joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
  joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
  joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
  joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
  effector_pub = rospy.Publisher("effector", Float64MultiArray, queue_size=10)
  k_pub = rospy.Publisher("k", Float64MultiArray, queue_size=10)
  bridge = CvBridge()

  state = {
    'q': None,
    'effector': None,
    'desired_joint_config': [
      np.array([0.0, 0.0, 0.0, 0.0]),
      np.array([1.0, 0.0, 0.0, 0.0]),  # No visiual difference to 0.0, 0.0, 0.0, 0.0
      np.array([1.0, 1.0, 0.0, 0.0]),
      np.array([1.0, 1.0, 0.0, 1.0]),
      np.array([0.0, 0.0, 0.0, 1.0]),
      np.array([0.0, 1.0, 0.0, 1.0]),
      np.array([1.0, 1.0, 0.0, 1.0]),
      np.array([2.0, 1.0, 0.0, 1.0]),
      np.array([2.0, np.pi/2, 0.0, np.pi/2]),
      np.array([-2.0, np.pi/2, 0.0, -np.pi/2]),
    ],
  }

  def camera_callback(data_1, data_2):
    try:
      image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
      image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')
      positions_and_angles = calc_positions_and_angles(image_1, image_2)
      state['q'] = positions_and_angles['q']
      state['effector'] = positions_and_angles['joint_centers']['red']
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

    try:
      print('-------------------')
      print('Moving to:', joint_config)
      joint1_pub.publish(Float64(data=joint_config[0]))
      joint2_pub.publish(Float64(data=joint_config[1]))
      joint3_pub.publish(Float64(data=joint_config[2]))
      joint4_pub.publish(Float64(data=joint_config[3]))
      rospy.sleep(3)

      if state['q'] is not None and state['effector'] is not None:
        k = calc_k(state['q'])
        print('Estimated q', state['q'])
        print('Estimated effector:', state['effector'])
        print('k', k)
        effector_pub.publish(Float64MultiArray(data=state['effector']))
        k_pub.publish(Float64MultiArray(data=k))
      else:
        print('Unable to determine angles / effector position')

    except Exception as ex:
      traceback.print_exc()

    rospy.Timer(rospy.Duration(1), move_robot, oneshot=True)

  rospy.Timer(rospy.Duration(3), move_robot, oneshot=True)
  rospy.spin()

if __name__ == '__main__':
    main()
