#!/usr/bin/env python

import cv2
import numpy as np
from scipy import optimize

def nhsv(*hsv):
  return (hsv[0] * 180/360, hsv[1] * 255/100, hsv[2] * 255/100)

colour_ranges = {
  'yellow': (
    (nhsv(59,0,0), nhsv(60,100,100)),
  ),
  'blue': (
    (nhsv(230,90,20), nhsv(240,100,100)),
  ),
  'green': (
    (nhsv(65,0,0), nhsv(125, 100, 100)),
  ),
  'red': (
    (nhsv(0,90,30), nhsv(4,100,100)),  # Red
    (nhsv(5,0,0), nhsv(20,100,100)),   # Red + orange
  ),
  'orange': (
    # Two ranges, since in the middle we have yellow
    (nhsv(35,0,0), nhsv(50,100,100)),     # Orange
    (nhsv(65,0,0), nhsv(100, 100, 100)),  # Orange + green
    (nhsv(5,0,0),  nhsv(20,100,100)),     # Red + orange
  )
}
dilate_kernel = np.ones((5, 5), np.uint8)


def calc_positions_and_angles(image_1, image_2):
  image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV)
  image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2HSV)

  def calc_center_of_masses(image, range_names):
    def calc_center_of_mass(colour_range):
      mask = np.zeros(image.shape[:2], image.dtype)
      for c_range in colour_range:
        mask = mask | cv2.inRange(image, *c_range)

      # Dilation gives a better center of mass calculation, since it partially compensates for
      # occlusion, as long there is only one source of the colour, and the real-world shape is
      # fairly regular
      mask = cv2.dilate(mask, dilate_kernel, iterations=3)
      M = cv2.moments(mask)
      return \
        np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]) if M['m00'] != 0.0 else \
        None

    return {
      range_name: calc_center_of_mass(colour_ranges[range_name])
      for range_name in range_names
    }

  def calc_circ_center(image, range_name):
    colour_range = colour_ranges[range_name]

    mask = np.zeros(image.shape[:2], image.dtype)
    for c_range in colour_range:
      mask = mask | cv2.inRange(image, *c_range)

    blur = cv2.GaussianBlur(mask, (7,7), 0)
    houghcircles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=10, param2=20, minRadius=8, maxRadius=17)

    return \
      None if houghcircles is None else \
      houghcircles[0,0,:2]

  def pixel_coords_to_meters_converter(meters_per_pixel, origin_pixels):
    def _coords_convert(coords_pixels):
      return \
        None if coords_pixels is None else \
        np.array([
          (coords_pixels[0] - origin_pixels[0]) * meters_per_pixel,
          (origin_pixels[1] - coords_pixels[1]) * meters_per_pixel  # Pixel y coords opposite to base frame
        ])

    return _coords_convert

  def coords_2d_to_3d(y_z, x_z):
    # For lack of anything better, we just average the z-coordates
    # Maybe could work with missing coordinates better further up the stack?
    return \
      None if x_z is None or y_z is None else \
      np.array([x_z[0], y_z[0], (y_z[1] + y_z[1]) / 2])

  # Find joint positions
  joint_range_names = ('yellow', 'blue', 'green', 'red')
  joint_centers_1 = calc_center_of_masses(image_1, joint_range_names)
  joint_centers_2 = calc_center_of_masses(image_2, joint_range_names)

  # If the joints are completely hidden, based on our knowledge of the structure
  # of the robot and positions of the cameras, we make some rough guesses
  if joint_centers_1['red'] is None and joint_centers_1['green'] is not None:
    joint_centers_1['red'] = joint_centers_1['green']
  if joint_centers_1['green'] is None and joint_centers_1['red'] is not None:
    joint_centers_1['green'] = joint_centers_1['red']
  if joint_centers_2['red'] is None and joint_centers_2['green'] is not None:
    joint_centers_2['red'] = joint_centers_2['green']
  if joint_centers_2['green'] is None and joint_centers_2['red'] is not None:
    joint_centers_2['green'] = joint_centers_2['red']

  # Find orange object positions
  target_center_1 = calc_circ_center(image_1, 'orange')
  target_center_2 = calc_circ_center(image_2, 'orange')

  # Use blue and yellow to convert from pixels to meters, since we know/assume they can't be
  # obscured, know the meter distance between them, and they can't move
  # We create conversions for both cameras, in case they are different
  meters_per_pixel_1 = 2.0 / (joint_centers_1['yellow'][1] - joint_centers_1['blue'][1])
  meters_per_pixel_2 = 2.0 / (joint_centers_2['yellow'][1] - joint_centers_2['blue'][1])

  # Convert all pixel coordinates to their meter coordinates
  # (The commented out part I think makes it world-coords, since the robot is 0 + 1.0m up)
  pixel_coords_to_meters_1 = pixel_coords_to_meters_converter(
    meters_per_pixel_1, joint_centers_1['yellow'] # + np.array([0, 1.0 / meters_per_pixel_1])
  )
  pixel_coords_to_meters_2 = pixel_coords_to_meters_converter(
    meters_per_pixel_2, joint_centers_2['yellow'] # + np.array([0, 1.0 / meters_per_pixel_2])
  )
  joint_centers_1 = {
    range_name: pixel_coords_to_meters_1(coords) for range_name, coords in joint_centers_1.items()
  }
  joint_centers_2 = {
    range_name: pixel_coords_to_meters_2(coords) for range_name, coords in joint_centers_2.items()
  }
  target_center_1 = pixel_coords_to_meters_1(target_center_1)
  target_center_2 = pixel_coords_to_meters_2(target_center_2)

  # Combine the 2D coordinates to 3D
  joint_centers = {
    range_name: coords_2d_to_3d(joint_centers_1[range_name], joint_centers_2[range_name]) for range_name in joint_range_names
  }
  target_center = coords_2d_to_3d(target_center_1, target_center_2)

  # Deliberatly don't compare to the estimate of blue. The estimate can only
  # be worse than reality
  link_1 = np.arctan2(joint_centers['green'][0], -joint_centers['green'][1])

  # We keep link 3 == 0, and keep link 2 +ve so all joints are in a plane,
  # rotated from the yz plane by link_1. So we rotate by -link_1 to then be
  # able to use atan2 to find the remaining link angles
  rotation = np.array([
    [np.cos(-link_1), -np.sin(-link_1), 0],
    [np.sin(-link_1), np.cos(-link_1), 0],
    [0, 0, 1],
  ])
  green_rot = rotation.dot(joint_centers['green'])
  red_rot = rotation.dot(joint_centers['red'])

  link_2 = np.arctan2(-green_rot[1], green_rot[2] - 2)
  link_3 = 0.0

  red_green_diff = red_rot - green_rot
  link_4 = np.arctan2(-red_green_diff[1], red_green_diff[2]) - link_2

  return {
    'joint_centers': joint_centers,
    'target_center': target_center,
    'q': np.array([link_1, link_2, link_3, link_4]),
  }


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


def calc_k_to_green(q):
  # Forward kinematics, but only to to the green joint, so we can get enough
  # equations to solve for the four angles numerically

  # cos and sin functions take 1-indexed as in mathematical notation
  def c(i):
    return np.cos(q[i - 1])

  def s(i):
    return np.sin(q[i - 1])

  return np.array([
    3*c(1)*s(3) + 3*c(3)*s(1)*s(2),
    -3*c(1)*c(3)*s(2) + 3*s(1)*s(3),
     3*c(2)*c(3) + 2,
  ])


def calc_jacobian(q):
  # cos and sin functions take 1-indexed as in mathematical notation
  def c(i):
    return np.cos(q[i - 1])

  def s(i):
    return np.sin(q[i - 1])

  return np.array([
    [
      2*c(1)*c(2)*s(4) + 3*c(1)*c(3)*s(2) + c(4)*(2*c(1)*c(3)*s(2) - 2*s(1)*s(3)) - 3*s(1)*s(3),
      2*c(2)*c(3)*c(4)*s(1) + 3*c(2)*c(3)*s(1) - 2*s(1)*s(2)*s(4),
      3*c(1)*c(3) + c(4)*(2*c(1)*c(3) - 2*s(1)*s(2)*s(3)) - 3*s(1)*s(2)*s(3),
      2*c(2)*c(4)*s(1) - s(4)*(2*c(1)*s(3) + 2*c(3)*s(1)*s(2)),
    ],
    [
      3*c(1)*s(3) + 2*c(2)*s(1)*s(4) + 3*c(3)*s(1)*s(2) + c(4)*(2*c(1)*s(3) + 2*c(3)*s(1)*s(2)),
      -2*c(1)*c(2)*c(3)*c(4) - 3*c(1)*c(2)*c(3) + 2*c(1)*s(2)*s(4),
      3*c(1)*s(2)*s(3) + 3*c(3)*s(1) + c(4)*(2*c(1)*s(2)*s(3) + 2*c(3)*s(1)),
      -2*c(1)*c(2)*c(4) - s(4)*(-2*c(1)*c(3)*s(2) + 2*s(1)*s(3)),
    ],
    [
      0,
      -2*c(2)*s(4) - 2*c(3)*c(4)*s(2) - 3*c(3)*s(2),
      -2*c(2)*c(4)*s(3) - 3*c(2)*s(3),
      -2*c(2)*c(3)*s(4) - 2*c(4)*s(2),
    ],
  ])


def calc_positions_and_angles_least_squares(image_1, image_2):
  pos_angles = calc_positions_and_angles(image_1, image_2)

  # Function to minimise
  def residual(q):
    return np.concatenate([
      pos_angles['joint_centers']['red'] - calc_k(q),
      pos_angles['joint_centers']['green'] - calc_k_to_green(q),
    ])

  q_0 = np.array([0.5, 0.5, 0.5, 0.5])
  # We constrain link 1 to 1/2 of its range to avoid visually close by
  # "far" solutions in terms of the angles
  bounds = (
    np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]),
    np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]),
  )
  result = optimize.least_squares(residual, q_0, bounds=bounds)

  return {
    'joint_centers': pos_angles['joint_centers'],
    'target_center': pos_angles['target_center'],
    'q': result.x,
  }
