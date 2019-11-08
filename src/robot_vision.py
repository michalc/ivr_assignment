#!/usr/bin/env python

import cv2
import numpy as np


colour_ranges = {
  # Ranges handle the slight shadow towards the bottom of the objects
  'yellow': ((0, 100, 100), (50, 255, 255)),
  'blue': ((100, 0, 0), (255, 50, 50)),
  'green': ((0, 100, 0), (50, 255, 50)),
  'red': ((0, 0, 100), (50, 50, 255)),
  'orange': ((75, 110, 130), (95, 175, 220)),
}
dilate_kernel = np.ones((5, 5), np.uint8)


def calc_positions_and_angles(image_1, image_2):

  def calc_center_of_masses(image, range_names):
    def calc_center_of_mass(colour_range):
      mask = cv2.inRange(image, *colour_range)
      # Dilation gives a better center of mass calculation, since it partially compensates for
      # occlusion, as long there is only one source of the colour, and the real-world shape is
      # fairly regular
      mask = cv2.dilate(mask, dilate_kernel, iterations=3)
      M = cv2.moments(mask)
      return np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])

    return {
      range_name: calc_center_of_mass(colour_ranges[range_name])
      for range_name in range_names
    }

  def calc_circ_centers(image, range_names):
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

    def calc_circ_center(colour_range):
      mask = cv2.inRange(image, *colour_range)
      mask = cv2.dilate(mask, dilate_kernel, iterations=3)
      _, _, stats, _ = cv2.connectedComponentsWithStats(mask)

      # Rough way to distinguish circle from rectangle if both quite visible: the rectangle will
      # have a larger proportion of pixels in its bounding box
      stats = stats[stats[:,cv2.CC_STAT_HEIGHT] != image.shape[0]]             # Remove background
      stats = stats[stats[:,cv2.CC_STAT_AREA].argsort()][-2:]                  # Take the largest two by area, in case of extra noise
      stats = stats[np.apply_along_axis(area_proportion, 1, stats).argsort()]  # Sort by proportion of area

      return \
        centre(stats[0]) if stats.shape[0] == 2 else \
        None

    return {
      range_name: calc_circ_center(colour_ranges[range_name])
      for range_name in range_names
    }

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

  # Find orange object positions
  target_center_1 = calc_circ_centers(image_1, ('orange',))['orange']
  target_center_2 = calc_circ_centers(image_2, ('orange',))['orange']

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

  green_to_red = joint_centers['red'] - joint_centers['green']
  blue_to_green = joint_centers['green'] - joint_centers['blue']
  yellow_to_blue = joint_centers['blue'] - joint_centers['yellow']
  green_to_red_norm = np.linalg.norm(green_to_red)
  blue_to_green_norm = np.linalg.norm(blue_to_green)
  yellow_to_blue_norm = np.linalg.norm(yellow_to_blue)

  # For simplicity and to deal with visually indistinguisable states, we enforce that link_3 is 0.0
  y_axis = np.array([0.0, 1.0])
  y_axis_norm = 1.0
  blue_x_y = np.array([joint_centers['blue'][0], joint_centers['blue'][1]])
  red_x_y = np.array([joint_centers['red'][0], joint_centers['red'][1]])
  red_to_blue = blue_x_y - red_x_y
  red_to_blue_norm = np.linalg.norm(red_to_blue)

  link_1 = \
    np.arccos(np.dot(red_to_blue, y_axis) / (red_to_blue_norm * y_axis_norm)) if red_to_blue_norm else \
    0.0  # If red to blue is completely vertical, we just assume no rotation
  link_2 = np.arccos(np.dot(blue_to_green, yellow_to_blue) / (blue_to_green_norm * yellow_to_blue_norm))
  link_3 = 0.0
  link_4 = np.arccos(np.dot(green_to_red, blue_to_green) / (green_to_red_norm * blue_to_green_norm))

  return {
    'joint_centers': joint_centers,
    'target_center': target_center,
    'q': np.array([link_1, link_2, link_3, link_4]),
  }
