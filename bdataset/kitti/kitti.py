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


from bdataset.kitti.common import Message, Pose
from bdataset.kitti.sensor import Lidar, Camera, IMU
from bdataset.kitti.geometry import Euler
from bdataset.kitti.proj_helper import latlon2utm
from modules.common_msgs.sensor_msgs import gnss_best_pose_pb2


class KITTISchema(object):
  """KITTI schema

  Args:
      object (_type_): _description_
  """
  def __init__(self, dataroot=None) -> None:
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

  def oxts_schemes(self):
    path_name = 'oxts'
    timestamps = self._read_timestamps(path_name)
    filenames = self._read_filenames(path_name)
    assert len(timestamps) == len(filenames)

    return [IMU(t, f) for t, f in zip(timestamps, filenames)]

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
        msg = Message(channel='pose', timestamp=oxts.timestamp, file_path=oxts.file_path, raw_data=ego_pose)
        self._messages.append(msg)
      # gnss, we faked some data
      if 'best_pose' in self._allowed_msgs:
        gnss_data = {
          'lat': oxts.lat,
          'lon': oxts.lon,
          'alt': oxts.alt,
          'undulation': 0,
          'latitude_std_dev': 0.05,
          'longitude_std_dev': 0.05,
          'height_std_dev': 0.05,
          'datum_id': gnss_best_pose_pb2.DatumId.WGS84,
          'sol_status': gnss_best_pose_pb2.SolutionStatus.SOL_COMPUTED,
          'sol_type': gnss_best_pose_pb2.SolutionType.NARROW_INT,
          'num_sats_tracked': int(oxts.numsats),
          'num_sats_l1': int(oxts.numsats),
          'num_sats_multi': int(oxts.numsats)
        }
        msg = Message(channel='best_pose', timestamp=oxts.timestamp, file_path=oxts.file_path, raw_data=gnss_data)
        self._messages.append(msg)
      # imu
      if 'imu' in self._allowed_msgs:
        linear_acc = [oxts.af, oxts.al, oxts.au]
        angular_vel = [oxts.wf, oxts.wl, oxts.wu]
        imu_data = {
          "linear_acceleration": linear_acc,
          "angular_velocity": angular_vel
        }
        msg = Message(channel='imu', timestamp=oxts.timestamp, file_path=oxts.file_path, raw_data=imu_data)
        self._messages.append(msg)
    # read camera
    if 'camera' in self._allowed_msgs:
      for camera_name, schemes in self._kitti_schema.camera_schemes().items():
        for camera in schemes:
          msg = Message(channel=camera_name, timestamp=camera.timestamp, file_path=camera.file_path)
          self._messages.append(msg)
    # sort by timestamp
    self._messages.sort(key=lambda msg : msg.timestamp)
