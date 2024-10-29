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
'''Generate apollo record file by kitti raw sensor data.'''

import logging
import math

from cyber_record.record import Record
from record_msg.builder import (
  ImageBuilder,
  PointCloudBuilder,
  LocalizationBuilder,
  TransformBuilder,
  IMUBuilder,
  GnssBestPoseBuilder)
from bdataset.kitti.kitti import KITTISchema, KITTI
from bdataset.kitti.geometry import Quaternion, Euler, rotation_matrix_to_euler
from bdataset.kitti.params import (
  kitti2apollo_lidar,
)

LOCALIZATION_TOPIC = '/apollo/localization/pose'
TF_TOPIC= '/tf'
IMU_TOPIC = '/apollo/sensor/gnss/imu'
GNSS_BEST_POSE_TOPIC = '/apollo/sensor/gnss/best_pose'

def dataset_to_record(kitti, record_root_path, gnss_hz=1):
  """Construct record message and save it as record

  Args:
      kitti (_type_): kitti
      record_root_path (str): record file saved path
  """
  image_builder = ImageBuilder()
  pc_builder = PointCloudBuilder()
  localization_builder = LocalizationBuilder()
  transform_builder = TransformBuilder()
  imu_builder = IMUBuilder()
  gnss_builder = GnssBestPoseBuilder()

  with Record(record_root_path, mode='w') as record:
    last_gnss_t = 0
    for msg in kitti:
      c, f, raw_data, t = msg.channel, msg.file_path, msg.raw_data, msg.timestamp
      logging.debug("{}, {}, {}, {}".format(c, f, raw_data, t))
      pb_msg = None
      # There're mix gray and rgb image files, so we just choose rgb image
      if c == 'image_02' or c == 'image_03':
        # KITTI image types: 'gray', 'bgr8'
        pb_msg = image_builder.build(f, 'camera', 'bgr8', t)
        channel_name = "/apollo/sensor/camera/{}/image".format(c)
        record.write(channel_name, pb_msg, int(t*1e9))
      elif c.startswith('velodyne'):
        pb_msg = pc_builder.build_nuscenes(f, 'velodyne', t, kitti2apollo_lidar, 255)
        channel_name = "/apollo/sensor/{}/compensator/PointCloud2".format(c)
        record.write(channel_name, pb_msg, int(t*1e9))
      elif c == "imu":
        # imu data
        pb_msg = imu_builder.build(raw_data['linear_acceleration'], 
                                   raw_data['angular_velocity'], t, 
                                   measurement_span=raw_data.get('measurement_span', 0))
        record.write(IMU_TOPIC, pb_msg, int(t*1e9))
      elif c == "best_pose":
        if t - last_gnss_t < 0.99:
          continue
        last_gnss_t = t
        # gnss best pose
        pos_args = [raw_data['lat'], raw_data['lon'], raw_data['alt'], raw_data['undulation'], t]
        kwargs = {}
        for k in raw_data.keys():
          if k in ['lat', 'lon', 'alt', 'undulation']:
            pass
          else:
            kwargs[k] = raw_data[k]
        pb_msg = gnss_builder.build(*pos_args, **kwargs)
        record.write(GNSS_BEST_POSE_TOPIC, pb_msg, int(t*1e9))
      elif c == "pose":
        # ego pose
        rotation = raw_data.rotation
        quat = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        heading = quat.to_euler().yaw

        # Apollo coordinate system conversion
        world_to_imu_q = Euler(0, 0, -math.pi/2).to_quaternion()
        quat *= world_to_imu_q

        pb_msg = localization_builder.build(
          raw_data.translation, [quat.w, quat.x, quat.y, quat.z], heading, t)
        if pb_msg:
          record.write(LOCALIZATION_TOPIC, pb_msg, int(t*1e9))

        pb_msg = transform_builder.build('world', 'localization',
          raw_data.translation, [quat.w, quat.x, quat.y, quat.z], t)
        if pb_msg:
          record.write(TF_TOPIC, pb_msg, int(t*1e9))


def convert_dataset(dataset_path, record_path, allowed_msgs=None, oxts_path='oxts'):
  """Generate apollo record file by KITTI dataset

  Args:
      dataset_path (str): KITTI dataset path
      record_path (str): record file saved path
  """
  kitti_schema = KITTISchema(dataroot=dataset_path, oxts_path=oxts_path)
  kitti = KITTI(kitti_schema, allowed_msgs=allowed_msgs)

  print("Start to convert scene, Pls wait!")
  dataset_to_record(kitti, record_path)
  print("Success! Records saved in '{}'".format(record_path))
