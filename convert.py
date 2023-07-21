# Convert pandaset to semantic kitti format
# Usage: python convert.py <path_to_pandaset> <path_to_output>

import os
import sys
import quaternion
import numpy as np
from pandaset import DataSet

print("Usage: python convert.py <path_to_pandaset> <path_to_output>")

dataset_path = sys.argv[1]
output_path = sys.argv[2]

# load Pandaset
dataset = DataSet(dataset_path)

# label mapping from pandaset to semantic kitti
# TODO: specify mapping
label_map = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 70,
    6: 72,
    7: 40,
    8: 60,
    9: 60,
    10: 49,
    11: 48,
    12: 49,
    13: 10,
    14: 18,
    15: 18,
    16: 18,
    17: 20,
    18: 15,
    19: 20,
    20: 20,
    21: 20,
    22: 20,
    23: 13,
    24: 20,
    25: 15,
    26: 11,
    27: 16,
    28: 20,
    29: 20,
    30: 30,
    31: 30,
    32: 0,
    33: 0,
    34: 99,
    35: 99,
    36: 81,
    37: 99,
    38: 81,
    39: 99,
    40: 99,
    41: 50,
    42: 99
}

print(f"Sequences found: {len(dataset.sequences())}")

for scene_id in dataset.sequences():
    print(f'Processing scene: {scene_id}')

    # extract Sequence object
    scene = dataset[scene_id]

    # skip if sequence does not contain annotations
    if scene.semseg is None:
        print('Does not contain annotations')
        continue

    scene.load_lidar()
    scene.load_semseg()

    # Create the SemanticKITTI label directory
    output_dir = os.path.join(output_path, "sequences", scene_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # use only 360° lidar
    scene.lidar.set_sensor(0)
    lidar_data = scene.lidar.data

    # retrieve sensor poses
    poses = scene.lidar.poses

    num_points = lidar_data[0].shape[0]

    # Convert each LiDAR frame in the scene (data is a list of dataframes)
    for index, lidar in enumerate(lidar_data):
        # Load LiDAR points and labels
        points = lidar.to_numpy()[:, :3]
        labels = scene.semseg[index].to_numpy()[:points.shape[0]]

        # print(points.shape)
        # print(labels.shape)
        assert points.shape[0] == labels.shape[0]

        # in case normalize z coordinates in kitti format
        # z_coords = points[:, 2]
        # Calculate the minimum and maximum values of the X-coordinate
        # z_min = np.min(z_coords)
        # z_max = np.max(z_coords)
        # Normalize the Z-coordinate values to the range (-3, 24)
        # normalized_z_coords = -6 + (z_coords - z_min) * (24 / (z_max - z_min))
        # Update the normalized X-coordinate values in the original array
        # points[:, 2] = normalized_z_coords

        # convert coordinates wrt LiDAR
        pose = poses[index]
        position = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
        orientation = list(pose['heading'].values())
        R = quaternion.as_rotation_matrix(np.quaternion(*orientation))

        # translate point cloud
        points -= position

        # apply transformation
        point_cloud = np.dot(points, R.T)

        # add intensity values
        point_cloud = np.column_stack((point_cloud, lidar.to_numpy()[:, 3]))

        # Convert Pandaset labels to SemanticKITTI labels
        semantic_labels = np.array([label_map[label[0]] for label in labels])

        # Save LiDAR points and labels in SemanticKITTI format
        output_file = os.path.join(output_dir, "velodyne", "{0:02d}".format(index) + ".bin")
        point_cloud.astype(np.float32).tofile(output_file)
        semantic_labels.astype(np.uint32).tofile(os.path.join(output_dir, "labels", "{0:02d}".format(index) + ".label"))

    # unload scene
    dataset.unload(scene_id)