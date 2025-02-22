#!/usr/bin/env python3

###############################################################################
# Copyright 2022 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import os
import sys
import logging

from bdataset.nuscenes.dataset_converter import convert_dataset as nuscenes_convert_dataset
from bdataset.nuscenes.calibration_converter import convert_calibration as nuscenes_convert_calibration
from bdataset.nuscenes.pcd_converter import convert_pcd as nuscenes_convert_pcd
from bdataset.kitti.dataset_converter import convert_dataset as kitti_convert_dataset
from bdataset.kitti.calibration_converter import convert_calibration as kitti_convert_calibration
from bdataset.kitti.pcd_converter import convert_pcd as kitti_convert_pcd
from bdataset.apolloscape.dataset_converter import convert_dataset as apolloscape_convert_dataset
from bdataset.apolloscape.calibration_converter import convert_calibration as apolloscape_convert_calibration
from bdataset.apolloscape.pcd_converter import convert_pcd as apolloscape_convert_pcd


def process_record(args):
  """Process record
  """
  # default output
  if args.output is None:
    if args.dataset == 'n':
      args.output = '.'
    elif args.dataset == 'k':
      args.output = 'result.record'

  if os.path.isdir(args.input):
    # choose dataset
    if args.dataset == 'n':
      nuscenes_convert_dataset(args.input, args.output)
    elif args.dataset == 'k':
      kitti_convert_dataset(args.input, args.output, args.oxts_path, args.lidar_sub_path, args.gnss_hz, args.allowed_msgs)
    elif args.dataset == 'a':
      apolloscape_convert_dataset(args.input, args.output)
    else:
      logging.error("Unsupported dataset type! '{}'".format(args.dataset))
  else:
    logging.error("Pls check the input directory! '{}'".format(args.input))

def process_calibration(args):
  """Process calibration
  """
  # default output
  if args.output is None:
    args.output = '.'

  if os.path.isdir(args.input):
    # choose dataset
    if args.dataset == 'n':
      nuscenes_convert_calibration(args.input, args.output)
    elif args.dataset == 'k':
      kitti_convert_calibration(args.input, args.output)
    elif args.dataset == 'a':
      apolloscape_convert_calibration(args.input, args.output)
    else:
      logging.error("Unsupported dataset type! '{}'".format(args.dataset))
  else:
    logging.error("Pls check the input directory! '{}'".format(args.input))

def process_pointcloud(args):
  """Process pointcloud
  """
  # default output
  if args.output is None:
    args.output = 'result.pcd'

  if os.path.isfile(args.input):
    # choose dataset
    if args.dataset == 'n':
      nuscenes_convert_pcd(args.input, args.output)
    elif args.dataset == 'k':
      kitti_convert_pcd(args.input, args.output)
    elif args.dataset == 'a':
      apolloscape_convert_pcd(args.input, args.output)
    else:
      logging.error("Unsupported dataset type! '{}'".format(args.dataset))
  else:
    logging.error("Pls check the input file! '{}'".format(args.input))


def main(args=sys.argv):
  """main
  """
  parser = argparse.ArgumentParser(
    description="Convert datasets (nuScenes, KITTI) to Apollo record files.",
    prog="main.py")

  parser.add_argument(
    "-d", "--dataset", action="store", type=str, required=True,
    choices=['n', 'k', 'a', 'w'],
    help="Dataset type. n:nuScenes, k:KITTI, w:Waymo")
  parser.add_argument(
    "-i", "--input", action="store", type=str, required=True,
    help="Input file or directory.")
  parser.add_argument(
    "-o", "--output", action="store", type=str, required=False,
    help="Output file or directory.")
  parser.add_argument(
    "-t", "--type", action="store", type=str, required=False,
    default="rcd", choices=['rcd', 'cal', 'pcd'],
    help="Conversion type. rcd:record, cal:calibration, pcd:pointcloud")
  parser.add_argument(
    "-m", "--allowed_msgs", action="store", type=str, required=False,
    nargs="+", choices=['velodyne64', 'imu', 'best_pose', 'pose', 'camera', 'lo_imu', 'lo_gnss'],
    default=None, help="The allowed_msgs in record file.")
  parser.add_argument(
    "-op", "--oxts_path", action="store", type=str, required=False,
    default='oxts.csv', help="The relative path of oxts.")
  parser.add_argument(
    "-lsp", "--lidar_sub_path", action="store", type=str, required=False,
    default='data', help="LiDAR data sub path")
  parser.add_argument(
    "-g", "--gnss_hz", action="store", type=int, required=False,
    default=1, help="The gnss frequency.")

  args = parser.parse_args(args[1:])
  logging.debug(args)

  if args.type == 'rcd':
    process_record(args)
  elif args.type == 'cal':
    process_calibration(args)
  elif args.type == 'pcd':
    process_pointcloud(args)
  else:
    logging.error("Input error! '{}'".format(args.input))


if __name__ == '__main__':
  main()
