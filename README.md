# LiDAR-Projection
ROS-Implementation of LiDAR 2d projection

## Result of Kitti Odometry Dataset
![seq07_54](https://github.com/soup1997/LiDAR-Projection/assets/86957779/07bffc60-3142-44e1-b255-a1f317c199aa)

## Usage
1. Download the kitti bag file from [kitti2bag](https://github.com/tomas789/kitti2bag)
2. Modify `args` indicating the path of the bag file in the launch file
```xml
<node pkg="rosbag" type="play" name="rosbag" args="-l your_bag_file_path"/>
```
3. Run the following command
```bash
roslaunch lidar_projection projection.launch
```
