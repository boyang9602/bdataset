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

import math

import numpy as np

from bdataset.kitti.common import Message, Pose
from bdataset.kitti.sensor import Lidar, Camera, OxTs, ExtendedOxTs, format_t
from bdataset.kitti.geometry import Euler
from bdataset.kitti.proj_helper import latlon2utm, utm2latlon, latlon2utmzone
from modules.common_msgs.sensor_msgs import gnss_best_pose_pb2

from bdataset.kitti.oxts2novatel import pos_mode2sol_type, pos_acc2std_dev

class KITTISchema(object):
  """KITTI schema

  Args:
      object (_type_): _description_
  """
  def __init__(self, dataroot=None, oxts_path='oxts.csv', lidar_path='data',
               imu_delay_path=None, gnss_delay_path=None, lidar_delay_path=None) -> None:
    self.lidar_path = lidar_path
    self.oxts_path = oxts_path
    self.imu_delay_path = imu_delay_path
    self.gnss_delay_path = gnss_delay_path
    self.lidar_delay_path = lidar_delay_path
    self.dataroot = dataroot
    self.camera_num = 4

  def lidar_schemes(self):
    path_name = 'velodyne_points'
    timestamps = self._read_timestamps(path_name)
    filenames = self._read_filenames(path_name, self.lidar_path)
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
      ('float_fields', 'f8', (30,)),
      ('int_fields', 'i1', (5,))
    ]
    data = np.loadtxt(os.path.join(self.dataroot, path_name), dtype=dtype, delimiter=',', skiprows=1)
    return [ExtendedOxTs(row[0], row[1].tolist() + row[2].tolist()) for row in data]
  
  def oxts_schemes(self):
    if self.oxts_path.startswith('extended'):
      return self.extended_oxts_schemes()
    return self.original_oxts_schemes()

  def imu_delays(self):
    if self.imu_delay_path is None:
      return None
    path = self.imu_delay_path
    delays = np.loadtxt(os.path.join(self.dataroot, path), dtype=np.float32)
    return delays

  def gnss_delays(self):
    if self.gnss_delay_path is None:
      return None
    path = self.gnss_delay_path
    delays = np.loadtxt(os.path.join(self.dataroot, path), dtype=np.float32)
    return delays

  def lidar_delays(self):
    if self.lidar_delay_path is None:
      return None
    path = self.lidar_delay_path
    delays = np.loadtxt(os.path.join(self.dataroot, path), dtype=np.float32)
    return delays

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
  def __init__(self, kitti_schema, allowed_msgs=None, warmup_time=0, undulation=False) -> None:
    self._kitti_schema = kitti_schema
    self._messages = []
    self._allowed_msgs = ['velodyne64', 'imu', 'best_pose', 'pose', 'camera'] if allowed_msgs is None else allowed_msgs
    self.warmup_time = warmup_time
    self.undulation = undulation
    self.read_messages()

  def __iter__(self):
    for message in self._messages:
      yield message

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def warmup_oxts_linear(self, oxts_list, warmup_time, frequency):
    first_oxts = oxts_list[0]
    g = 9.80983
    utm_zone = latlon2utmzone(first_oxts.lat, first_oxts.lon)
    values_list = []
    # calculate velocities at east and north
    tens = []
    for oxts in oxts_list:
      e, n, _ = latlon2utm(oxts.lat, oxts.lon)
      tens.append((oxts.timestamp, e, n))
    tens = np.array(tens)
    d_tens = np.diff(tens, axis=0)
    ves = d_tens[:, 1] / d_tens[:, 0]
    vns = d_tens[:, 2] / d_tens[:, 0]
    avg_ve = np.sum(ves) / len(ves)
    avg_vn = np.sum(vns) / len(vns)
    avg_vx = math.sqrt(avg_ve * avg_ve + avg_vn * avg_vn)
    fields = OxTs.fields if isinstance(first_oxts, OxTs) else ExtendedOxTs.fields
    init_t = first_oxts.timestamp
    init_east = tens[0, 1].item()
    init_north = tens[0, 2].item()
    for t in np.arange(first_oxts.timestamp - warmup_time, first_oxts.timestamp, step=1/frequency):
      values = [t]
      east = init_east - (init_t - t) * avg_ve
      north = init_north - (init_t - t) * avg_vn
      lat, lon = utm2latlon(east, north, utm_zone)
      zero_fields = ['pitch', 'vl', 'vu', 'ax', 'ay', 'af', 'al', \
                      'wx', 'wy', 'wz', 'wf', 'wl', 'wu']
      copy_fields = ['alt', 'height_msl', 'undulation', \
                      'roll', 'yaw', \
                      'pos_accuracy', 'vel_accuracy', \
                      'latitude_std_dev', 'longitude_std_dev', 'height_std_dev', \
                      'navstat', 'numsats', 'posmode', 'velmode', 'orimode']
      for field in fields:
        if field == 'lat':
          values.append(lat)
        elif field == 'lon':
          values.append(lon)
        elif field == 've':
          values.append(avg_ve)
        elif field == 'vn':
          values.append(avg_vn)
        elif field == 'vf':
          values.append(avg_vx)
        elif field in ['au', 'az']:
          values.append(g)
        elif field in zero_fields:
          values.append(0)
        elif field in copy_fields:
          values.append(getattr(oxts_list[0], field))
        else:
          raise RuntimeError('unknown field!')
      values_list.append(values)
    return values_list

  def get_gnss_n_imu_delays(self, num_oxts, num_warmup_oxts):
    gnss_delays = self._kitti_schema.gnss_delays()
    imu_delays = self._kitti_schema.imu_delays()
    if gnss_delays is None:
      gnss_delays = np.zeros(num_oxts)
    if imu_delays is None:
      imu_delays = np.zeros(num_oxts)
    warmup_gnss_delays = np.zeros(num_warmup_oxts) + np.sum(gnss_delays[:10]) / 10
    warmup_imu_delays = np.zeros(num_warmup_oxts) + np.sum(imu_delays[:10]) / 10
    full_gnss_delays = np.concatenate([warmup_gnss_delays, gnss_delays])
    full_imu_delays = np.concatenate([warmup_imu_delays, imu_delays])
    return full_gnss_delays, full_imu_delays

  def get_lidar_delays(self, num_lidar, num_warmup_lidar):
    lidar_delays = self._kitti_schema.lidar_delays()
    if lidar_delays is None:
      lidar_delays = np.zeros(num_lidar)
    warmup_lidar_delays = np.zeros(num_warmup_lidar) + np.sum(lidar_delays[:10]) / 10
    full_lidar_delays = np.concatenate([warmup_lidar_delays[-10:], lidar_delays])
    return full_lidar_delays

  def read_messages(self):
    oxts_schemes = self._kitti_schema.oxts_schemes()
    first_oxts = oxts_schemes[0]
    start_time = first_oxts.timestamp
    oxts_frequency = 100
    interval = 1 / oxts_frequency
    warmup_start_time = start_time - self.warmup_time
    warmup_oxts_schemes = []
    warmup_oxts_values = self.warmup_oxts_linear(oxts_schemes[:10], self.warmup_time, oxts_frequency)
    for i in range(oxts_frequency * self.warmup_time):
      warmup_oxts_schemes.append(type(first_oxts)(format_t(warmup_oxts_values[i][0]), warmup_oxts_values[i][1:]))
    gnss_delays, imu_delays = self.get_gnss_n_imu_delays(len(oxts_schemes), len(warmup_oxts_schemes))
    # read lidar
    if 'velodyne64' in self._allowed_msgs:
      lidar_schemes = self._kitti_schema.lidar_schemes()
      for i, lidar in enumerate(lidar_schemes):
        if lidar.timestamp < start_time:
          continue
        break
      lidar_schemes = lidar_schemes[i:]
      lidar_frequency = 10
      interval = 1 / lidar_frequency
      warmup_lidar_schemes = []
      first_lidar = lidar_schemes[0]
      for i in range(lidar_frequency * self.warmup_time):
        measurement_time = format_t(warmup_start_time + i * interval)
        warmup_lidar_schemes.append(Lidar(measurement_time, first_lidar.file_path))
      lidar_delays = self.get_lidar_delays(len(lidar_schemes), len(warmup_lidar_schemes))
      for i, lidar in enumerate(warmup_lidar_schemes[-10:] + lidar_schemes):
        measurement_time = lidar.timestamp
        raw_data = {
          'measurement_time': measurement_time
        }
        msg_time = measurement_time + lidar_delays[i]
        msg = Message(channel='velodyne64', timestamp=msg_time,
                      file_path=lidar.file_path, raw_data=raw_data)
        self._messages.append(msg)
    # read oxts
    for i, oxts in enumerate(warmup_oxts_schemes + oxts_schemes):
      # pose
      if 'pose' in self._allowed_msgs:
        ego_pose = Pose()
        utm_x, utm_y, _ = latlon2utm(oxts.lat, oxts.lon)
        ego_pose.set_translation(utm_x, utm_y, 0)
        euler = Euler(oxts.roll, oxts.pitch, oxts.yaw)
        q = euler.to_quaternion()
        ego_pose.set_rotation(q.w, q.x, q.y, q.z)
        measurement_time = oxts.timestamp
        raw_data = {
          'ego_pose': ego_pose,
          'measurement_time': measurement_time
        }
        msg_time = measurement_time
        msg = Message(channel='pose', timestamp=msg_time, raw_data=raw_data)
        self._messages.append(msg)
      # gnss, we faked some data
      if 'best_pose' in self._allowed_msgs:
        if isinstance(oxts, ExtendedOxTs):
          if self.undulation:
            height_msl = oxts.height_msl
            undulation = oxts.undulation
          else:
            height_msl = oxts.alt
            undulation = 0
          latitude_std_dev = oxts.latitude_std_dev
          longitude_std_dev = oxts.longitude_std_dev
          height_std_dev = oxts.height_std_dev
        else:
          height_msl = oxts.alt
          undulation = 0
          latitude_std_dev = pos_acc2std_dev(oxts.pos_accuracy)
          longitude_std_dev = pos_acc2std_dev(oxts.pos_accuracy)
          height_std_dev = pos_acc2std_dev(oxts.pos_accuracy)
        measurement_time = oxts.timestamp
        gnss_data = {
          'lat': oxts.lat,
          'lon': oxts.lon,
          'height_msl': height_msl,
          'undulation': undulation,
          'latitude_std_dev': latitude_std_dev,
          'longitude_std_dev': longitude_std_dev,
          'height_std_dev': height_std_dev,
          'datum_id': gnss_best_pose_pb2.DatumId.WGS84,
          'sol_status': gnss_best_pose_pb2.SolutionStatus.SOL_COMPUTED,
          'sol_type': pos_mode2sol_type(int(oxts.posmode))[1],
          'num_sats_tracked': int(oxts.numsats),
          'num_sats_in_solution': int(oxts.numsats),
          'num_sats_l1': int(oxts.numsats),
          'num_sats_multi': int(oxts.numsats),
          "measurement_time": measurement_time
        }
        msg_time = measurement_time + gnss_delays[i]
        msg = Message(channel='best_pose', timestamp=msg_time, raw_data=gnss_data)
        self._messages.append(msg)
      # imu
      if 'imu' in self._allowed_msgs:
        measurement_time = oxts.timestamp
        imu_data = {
          "ax": oxts.ax,
          "ay": oxts.ay,
          "az": oxts.az,
          "wx": oxts.wx,
          "wy": oxts.wy,
          "wz": oxts.wz,
          "measurement_time": measurement_time,
          "measurement_span": 0,
        }
        msg_time = measurement_time + imu_delays[i]
        msg = Message(channel='imu', timestamp=msg_time, raw_data=imu_data)
        self._messages.append(msg)
    # read camera
    if 'camera' in self._allowed_msgs:
      for camera_name, schemes in self._kitti_schema.camera_schemes().items():
        for camera in schemes:
          if camera.timestamp < start_time:
            continue
          measurement_time = camera.timestamp
          raw_data = {
            'measurement_time': measurement_time
          }
          msg_time = measurement_time
          msg = Message(channel=camera_name, timestamp=msg_time, file_path=camera.file_path, raw_data=raw_data)
          self._messages.append(msg)

    # sort by timestamp
    self._messages.sort(key=lambda msg : msg.timestamp)
