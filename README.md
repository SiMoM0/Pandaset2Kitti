# PandaSet 2 Kitti

Convert [Pandaset](https://pandaset.org/) to [SemanticKitti](http://www.semantic-kitti.org/) format.

## Dataset

Visit the [Pandaset](https://pandaset.org/) website, sign up and then download the dataset.

### Structure

#### Files & Folders

```text
.
├── LICENSE.txt
├── annotations
│   ├── cuboids
│   │   ├── 00.pkl.gz
│   │   .
│   │   .
│   │   .
│   │   └── 79.pkl.gz
│   └── semseg  // Semantic Segmentation is available for specific scenes
│       ├── 00.pkl.gz
│       .
│       .
│       .
│       ├── 79.pkl.gz
│       └── classes.json
├── camera
│   ├── back_camera
│   │   ├── 00.jpg
│   │   .
│   │   .
│   │   .
│   │   ├── 79.jpg
│   │   ├── intrinsics.json
│   │   ├── poses.json
│   │   └── timestamps.json
│   ├── front_camera
│   │   └── ...
│   ├── front_left_camera
│   │   └── ...
│   ├── front_right_camera
│   │   └── ...
│   ├── left_camera
│   │   └── ...
│   └── right_camera
│       └── ...
├── lidar
│   ├── 00.pkl.gz
│   .
│   .
│   .
│   ├── 79.pkl.gz
│   ├── poses.json
│   └── timestamps.json
└── meta
    ├── gps.json
    └── timestamps.json
```

## Requirements

Python packages required:
* Numpy: `pip install numpy`
* Numpy Quaternion: `pip install numpy-quaternion`
* Pandaset-devkit (install from the [repo](https://github.com/scaleapi/pandaset-devkit))

## Usage

Run the `convert.py` script secifying the path to Pandaset and the output path for the converted dataset as follow:

    python convert.py <path_to_pandaset> <path_to_output>

**Warning**: The script will generate only scenes that have semantic segmentation labels available, all the other will be skipped.