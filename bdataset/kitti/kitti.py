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

import os

import numpy as np

from bdataset.kitti.common import Message, Pose
from bdataset.kitti.sensor import Lidar, Camera, OxTs, ExtendedOxTs
from bdataset.kitti.geometry import Euler
from bdataset.kitti.proj_helper import latlon2utm
from modules.common_msgs.sensor_msgs import gnss_best_pose_pb2

from bdataset.kitti.oxts2novatel import pos_mode2sol_type, pos_acc2std_dev

class KITTISchema(object):
  """KITTI schema

  Args:
      object (_type_): _description_
  """
  def __init__(self, dataroot=None, oxts_path='oxts.csv') -> None:
    self.oxts_path = oxts_path
    self.dataroot = dataroot
    self.camera_num = 4

  def lidar_schemes(self):
    path_name = 'velodyne_points'
    timestamps = self._read_timestamps(path_name)
    filenames = self._read_filenames(path_name)
    assert len(timestamps) == len(filenames)

    return [Lidar(t, f) for t, f in zip(timestamps, filenames)]

  def camera_schemes(self):
    schemes = dict()
    for camera_id in range(self.camera_num):
      path_name = 'image_{:02d}'.format(camera_id)
      timestamps = self._read_timestamps(path_name)
      filenames = self._read_filenames(path_name)
      assert len(timestamps) == len(filenames)

      schemes[path_name] = [Camera(t, f) for t, f in zip(timestamps, filenames)]
    return schemes

  def original_oxts_schemes(self):
    path_name = self.oxts_path
    dtype = [
      ('timestamp', 'U29'),
      ('float_fields', 'f8', (25,)),
      ('int_fields', 'i1', (5,))
    ]
    data = np.loadtxt(os.path.join(self.dataroot, path_name), dtype=dtype, delimiter=',', skiprows=1)
    return [OxTs(row[0], row[1].tolist() + row[2].tolist()) for row in data]
  
  def extended_oxts_schemes(self):
    path_name = self.oxts_path
    dtype = [
      ('timestamp', 'U29'),
      ('float_fields', 'f8', (27,)),
      ('int_fields', 'i1', (5,))
    ]
    data = np.loadtxt(os.path.join(self.dataroot, path_name), dtype=dtype, delimiter=',', skiprows=1)
    return [ExtendedOxTs(row[0], row[1].tolist() + row[2].tolist()) for row in data]
  
  def oxts_schemes(self):
    if self.oxts_path.startswith('extended'):
      return self.extended_oxts_schemes()
    return self.original_oxts_schemes()

  def _read_timestamps(self, file_path, file_name='timestamps.txt'):
    timestamps_file = os.path.join(self.dataroot, file_path, file_name)
    timestamps = []
    with open(timestamps_file, 'r') as f:
      for line in f:
        timestamps.append(line.strip())
    return timestamps

  def _read_filenames(self, file_path, sub_path='data'):
    filenames = []
    absolute_path = os.path.join(self.dataroot, file_path, sub_path)
    for f in os.listdir(absolute_path):
      file_name = os.path.join(absolute_path, f)
      if os.path.isfile(file_name):
        filenames.append(file_name)
    # Need sorted by name
    filenames.sort()
    return filenames


class KITTI(object):
  """KITTI dataset

  Args:
      object (_type_): _description_
  """
  def __init__(self, kitti_schema, allowed_msgs=None) -> None:
    self._kitti_schema = kitti_schema
    self._messages = []
    self._allowed_msgs = ['velodyne64', 'imu', 'best_pose', 'pose', 'camera'] if allowed_msgs is None else allowed_msgs
    self.read_messages()

  def __iter__(self):
    for message in self._messages:
      yield message

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def read_messages(self):
    # read lidar
    if 'velodyne64' in self._allowed_msgs:
      for lidar in self._kitti_schema.lidar_schemes():
        msg = Message(channel='velodyne64', timestamp=lidar.timestamp, file_path=lidar.file_path)
        self._messages.append(msg)
    # read oxts
    for oxts in self._kitti_schema.oxts_schemes():
      # pose
      if 'pose' in self._allowed_msgs:
        ego_pose = Pose()
        utm_x, utm_y, _ = latlon2utm(oxts.lat, oxts.lon)
        ego_pose.set_translation(utm_x, utm_y, 0)
        euler = Euler(oxts.roll, oxts.pitch, oxts.yaw)
        q = euler.to_quaternion()
        ego_pose.set_rotation(q.w, q.x, q.y, q.z)
        msg = Message(channel='pose', timestamp=oxts.timestamp, raw_data=ego_pose)
        self._messages.append(msg)
      # gnss, we faked some data
      if 'best_pose' in self._allowed_msgs:
        gnss_data = {
          'lat': oxts.lat,
          'lon': oxts.lon,
          'height_msl': oxts.height_msl if isinstance(oxts, ExtendedOxTs) else oxts.alt,
          'undulation': oxts.undulation if isinstance(oxts, ExtendedOxTs) else 0,
          'latitude_std_dev': pos_acc2std_dev(oxts.pos_accuracy),
          'longitude_std_dev': pos_acc2std_dev(oxts.pos_accuracy),
          'height_std_dev': pos_acc2std_dev(oxts.pos_accuracy),
          'datum_id': gnss_best_pose_pb2.DatumId.WGS84,
          'sol_status': gnss_best_pose_pb2.SolutionStatus.SOL_COMPUTED,
          'sol_type': pos_mode2sol_type(int(oxts.posmode))[1],
          'num_sats_tracked': int(oxts.numsats),
          'num_sats_l1': int(oxts.numsats),
          'num_sats_multi': int(oxts.numsats)
        }
        msg = Message(channel='best_pose', timestamp=oxts.timestamp, raw_data=gnss_data)
        self._messages.append(msg)
      # imu
      if 'imu' in self._allowed_msgs:
        linear_acc = [oxts.ax, oxts.ay, oxts.az]
        angular_vel = [oxts.wx, oxts.wy, oxts.wz]
        imu_data = {
          "linear_acceleration": linear_acc,
          "angular_velocity": angular_vel
        }
        msg = Message(channel='imu', timestamp=oxts.timestamp, raw_data=imu_data)
        self._messages.append(msg)
    # read camera
    if 'camera' in self._allowed_msgs:
      for camera_name, schemes in self._kitti_schema.camera_schemes().items():
        for camera in schemes:
          msg = Message(channel=camera_name, timestamp=camera.timestamp, file_path=camera.file_path)
          self._messages.append(msg)

    # sort by timestamp
    self._messages.sort(key=lambda msg : msg.timestamp)
