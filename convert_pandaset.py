# Convert PandaSet dataset to KITTI format and add laser ID to point cloud.
# Create point cloud with attributes ['x', 'y', 'z', 'i', 'laser_id'], saved in .bin format and labels as .labels.
# Usage: python3 convert_pandaset.py --dataset /path/to/pandaset --raw /path/to/pandar64 --output_path /path/to/converted_dataset

import os
import argparse
import numpy as np
import json
import pickle
import gzip
import tqdm as tqdm
from scipy.spatial.transform import Rotation

# label names from pandaset
label_names = {
    1: 'Smoke',
    2: 'Exhaust',
    3: 'Spray or rain',
    4: 'Reflection',
    5: 'Vegetation',
    6: 'Ground',
    7: 'Road',
    8: 'Lane Line Marking',
    9: 'Stop Line Marking',
    10: 'Other Road Marking',
    11: 'Sidewalk',
    12: 'Driveway',
    13: 'Car',
    14: 'Pickup Truck',
    15: 'Medium-sized Truck',
    16: 'Semi-truck',
    17: 'Towed Object',
    18: 'Motorcycle',
    19: 'Other Vehicle - Construction Vehicle',
    20: 'Other Vehicle - Uncommon',
    21: 'Other Vehicle - Pedicab',
    22: 'Emergency Vehicle',
    23: 'Bus',
    24: 'Personal Mobility Device',
    25: 'Motorized Scooter',
    26: 'Bicycle',
    27: 'Train',
    28: 'Trolley',
    29: 'Tram / Subway',
    30: 'Pedestrian',
    31: 'Pedestrian with Object',
    32: 'Animals - Bird',
    33: 'Animals - Other',
    34: 'Pylons',
    35: 'Road Barriers',
    36: 'Signs',
    37: 'Cones',
    38: 'Construction Signs',
    39: 'Temporary Construction Barriers',
    40: 'Rolling Containers',
    41: 'Building',
    42: 'Other Static Object'
}

############### PARAMETERS ###############
FILE_EXTENSION = '.pkl.gz'
##########################################

def transform_pc(pc, pose):
    # get pose
    position = np.array(list(pose['position'].values()), dtype=np.float32)
    heading = np.array(list(pose['heading'].values()), dtype=np.float32)

    # convert quaternion to rotation matrix
    w, x, y, z = heading
    R = Rotation.from_quat([x, y, z, w]).as_matrix()

    # transformation matrix 4x4
    T = np.zeros(shape=(4, 4), dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = position
    T[3, 3] = 1

    # inverse transformation matrix
    T_inv = np.linalg.inv(T)

    # homogeneous coordinates
    hpoints = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))
    out_pc = np.matmul(T_inv, hpoints.T).T[:, :3]

    # attach other features
    out_pc = np.hstack((out_pc, pc[:, 3:]))

    return out_pc

def get_scan(lidar_path, labels_path, pose):
    with gzip.open(lidar_path, 'rb') as f:
        pc = pickle.load(f)
        # select only 360 degrees lidar
        where = pc['d'] == 0
        pc = pc[where]
        pc = pc.to_numpy(dtype=np.float32) # attributes ['x', 'y', 'z', 'i', 't']
    
    with gzip.open(labels_path, 'rb') as f:
        labels = pickle.load(f)
        labels = labels[where]
        labels = labels.to_numpy(dtype=np.uint32).reshape((-1, 1))
    
    assert pc.shape[0] == labels.shape[0], 'Scan and labels have different number of points'

    # apply transformation to point cloud
    pc = transform_pc(pc, pose)

    return pc, labels

def get_laserid(lidar_path):
    with gzip.open(lidar_path, 'rb') as f:
        pc = pickle.load(f)
        laser_id = pc['laser_id'].to_numpy(dtype=np.float32).reshape((-1, 1))
    
    return laser_id

def parse_args():
    parser = argparse.ArgumentParser(description='Modify Pandaset dataset.')
    parser.add_argument('--dataset', type=str, help='Path to the PandaSet dataset')
    parser.add_argument('--raw', type=str, help='Path to pandar64 raw data')
    parser.add_argument('--output_path', type=str, help='Path to save the converted dataset')
    return parser.parse_args()

def main(dataset_path, raw_data, output_path):
    # create output folder
    os.makedirs(output_path, exist_ok=True)

    # list all sequences
    sequences = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    print(f'Number of sequences: {len(sequences)}')
    # all sequences with semantic labels
    sem_sequences = [s for s in sequences if os.path.exists(os.path.join(dataset_path, s, 'annotations', 'semseg'))]
    print(f'With semantic labels: {len(sem_sequences)}')

    for s in sem_sequences:
        print(f'Processing sequence {s}', end=' ')
        seq_path = os.path.join(dataset_path, s)
        lidar_path = os.path.join(seq_path, 'lidar')
        labels_path = os.path.join(seq_path, 'annotations', 'semseg')
        poses_file = os.path.join(lidar_path, 'poses.json')
        raw_seq = os.path.join(raw_data, s)

        # gather all files names
        files = sorted([f for f in os.listdir(lidar_path) if f.endswith(FILE_EXTENSION)])
        labels = sorted([f for f in os.listdir(labels_path) if f.endswith(FILE_EXTENSION)])
        raw_lidar = sorted([f for f in os.listdir(raw_seq) if f.endswith(FILE_EXTENSION)])

        assert len(files) == len(labels), f'Lidar and Labels in sequence {s} have different number of files {len(files)} != {len(labels)}'
        assert len(files) == len(raw_lidar), f'Lidar and Raw in sequence {s} have different number of files {len(files)} != {len(raw_lidar)}'

        # extract poses
        with open(poses_file, 'r') as f:
            poses = json.load(f)

        # create output folder
        output_seq = os.path.join(output_path, s)
        os.makedirs(output_seq, exist_ok=True)

        # make lidar and labels folders
        output_lidar_path = os.path.join(output_seq, 'velodyne')
        os.makedirs(output_lidar_path, exist_ok=True)
        output_labels_path = os.path.join(output_seq, 'labels')
        os.makedirs(output_labels_path, exist_ok=True)

        for i, (f, l, r) in enumerate(zip(files, labels, raw_lidar)):
            lidar_file = os.path.join(lidar_path, f)
            labels_file = os.path.join(labels_path, l)
            raw_file = os.path.join(raw_seq, r)

            pc, labels = get_scan(lidar_file, labels_file, poses[i])
            laser_id = get_laserid(raw_file)

            # keep only x, y, z, i
            pc = pc[:, :4]

            assert pc.shape[0] == labels.shape[0], 'Scan and labels have different number of points'
            assert pc.shape[0] == laser_id.shape[0], 'Scan and laser_id have different number of points'

            # concatenate laser_id to point cloud --> ['x', 'y', 'z', 'i', 'laser_id']
            pc = np.hstack((pc, laser_id), dtype=np.float32)

            # save point cloud
            pc_file = os.path.join(output_lidar_path, '{:02}.bin'.format(i))
            pc.tofile(pc_file)

            # test loading point cloud
            #pc = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 5))

            # save labels
            labels_file = os.path.join(output_labels_path, '{:02}.label'.format(i))
            labels.tofile(labels_file)

        print(f'--> converted {len(files)} files')

if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset
    raw_data = args.raw
    output_path = args.output_path

    # Add your evaluation code here
    print('INFO')
    print('Dataset path: ', dataset_path)
    print('Raw data: ', raw_data)
    print('Output path: ', output_path)

    main(dataset_path, raw_data, output_path)