#!/usr/bin/env python

import sys
import traceback

import message_filters
import numpy as np
import rospy
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image


def main():
  rospy.init_node('error', anonymous=True)

  errors_x = []
  errors_y = []
  errors_z = []

  def error_callback(k, x, y, z):
    try:
      print '.',
      sys.stdout.flush()
      errors_x.append(np.absolute(k.data[0] - x.data))
      errors_y.append(np.absolute(k.data[1] - y.data))
      errors_z.append(np.absolute(k.data[1] - z.data))
    except Exception:
      traceback.print_exc()

  k_sub = message_filters.Subscriber('/k', Float64MultiArray)
  target_x_sub = message_filters.Subscriber('/target/x_position_controller/command', Float64)
  target_y_sub = message_filters.Subscriber('/target/y_position_controller/command', Float64)
  target_z_sub = message_filters.Subscriber('/target/z_position_controller/command', Float64)

  message_filters \
    .ApproximateTimeSynchronizer([k_sub, target_x_sub, target_y_sub, target_z_sub], queue_size=1, slop=0.01, allow_headerless=True) \
    .registerCallback(error_callback)

  start_time = rospy.get_time()

  rospy.spin()

  end_time = rospy.get_time()
  print('\nDuration: ', end_time - start_time)

  errors_x = np.array(errors_x)
  errors_y = np.array(errors_y)
  errors_z = np.array(errors_z)

  print('x-min:    ', np.min(errors_x))
  print('x-average:', np.average(errors_x))
  print('x-max:    ', np.max(errors_x))
  print('x-std:    ', np.std(errors_x))

  print('y-min:    ', np.min(errors_y))
  print('y-average:', np.average(errors_y))
  print('y-max:    ', np.max(errors_y))
  print('y-std:    ', np.std(errors_y))

  print('z-min:    ', np.min(errors_z))
  print('z-average:', np.average(errors_z))
  print('z-max:    ', np.max(errors_z))
  print('z-std:    ', np.std(errors_z))

if __name__ == '__main__':
    main()
