#!/usr/bin/env python

import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

from robot_vision import calc_positions_and_angles


def main():
  rospy.init_node('robot_vision_join_state', anonymous=True)
  joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
  joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
  joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
  joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
  bridge = CvBridge()

  state = {
    'q': None,
    'angles_command': [
      np.array([1.0, 1.0, 1.0, 1.0]),
    ],
  }

  def camera_callback(data_1, data_2):
    image_1 = bridge.imgmsg_to_cv2(data_1, 'bgr8')
    image_2 = bridge.imgmsg_to_cv2(data_2, 'bgr8')
    state['q'] = calc_positions_and_angles(image_1, image_2)['q']

  camera_1_sub = message_filters.Subscriber('/camera1/robot/image_raw', Image)
  camera_2_sub = message_filters.Subscriber('/camera2/robot/image_raw', Image)
  message_filters \
    .ApproximateTimeSynchronizer([camera_1_sub, camera_2_sub], queue_size=1, slop=0.01) \
    .registerCallback(camera_callback)

  def move_robot(_):
    try:
      angles = state['angles_command'].pop()
    except IndexError:
      rospy.core.signal_shutdown('No more angles')
      return

    print('Moving to:', angles)
    joint1_pub.publish(Float64(data=angles[0]))
    joint2_pub.publish(Float64(data=angles[1]))
    joint3_pub.publish(Float64(data=angles[2]))
    joint4_pub.publish(Float64(data=angles[3]))
    rospy.sleep(3)
    print('Estimated angles:', state['q'])

    rospy.Timer(rospy.Duration(1), move_robot, oneshot=True)

  rospy.Timer(rospy.Duration(3), move_robot, oneshot=True)
  rospy.spin()

if __name__ == '__main__':
    main()
