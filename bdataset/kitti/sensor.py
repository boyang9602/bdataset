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

from datetime import datetime


def to_timestamp(sensor_time):
  date_sec, nano_sec = sensor_time.split('.')
  time_sec = datetime.strptime(date_sec, '%Y-%m-%d %H:%M:%S')
  return datetime.timestamp(time_sec) + float(nano_sec)*1e-9

"""
Two kinds of sensors:
1. Sensors with binary data, like pictures or point clouds
2. Sensors with text data, like IMU
"""
class BaseSensor(object):
  def __init__(self):
    pass

class BinarySensor(BaseSensor):
  def __init__(self, timestamp, file_path):
    super().__init__()
    self.timestamp = to_timestamp(timestamp)
    self.file_path = file_path

class Lidar(BinarySensor):
  def __init__(self, timestamp, file_path) -> None:
    super().__init__(timestamp, file_path)

class Camera(BinarySensor):
  def __init__(self, timestamp, file_path) -> None:
    super().__init__(timestamp, file_path)

class TextSensor(BaseSensor):
  fields = None
  def __init__(self, timestamp, values: list):
    super().__init__()
    self.timestamp = timestamp
    self.parse(values)

  def parse(self, values):
    if OxTs.fields is None:
      raise RuntimeError("fields have to be declared!")
    assert len(OxTs.fields) == len(values), f"required fields ({len(OxTs.fields)}) and provided values ({len(values)}) have different lengths!"
    for field, value in zip(OxTs.fields, values):
      setattr(self, field, value)

class OxTs(TextSensor):
  fields = [
    'lat', 'lon', 'alt',
    'roll', 'pitch', 'yaw',
    'vn', 've', 'vf', 'vl', 'vu',
    'ax', 'ay', 'az', 'af', 'al', 'au',
    'wx', 'wy', 'wz', 'wf', 'wl', 'wu',
    'pos_accuracy', 'vel_accuracy',
    'navstat', 'numsats', 
    'posmode', 'velmode', 'orimode'
  ]

class ExtendedOxTs(TextSensor):
  fields = [
    'lat', 'lon', 'alt', 'height_msl', 'undulation',
    'roll', 'pitch', 'yaw',
    'vn', 've', 'vf', 'vl', 'vu',
    'ax', 'ay', 'az', 'af', 'al', 'au',
    'wx', 'wy', 'wz', 'wf', 'wl', 'wu',
    'pos_accuracy', 'vel_accuracy',
    'navstat', 'numsats', 
    'posmode', 'velmode', 'orimode'
  ]
