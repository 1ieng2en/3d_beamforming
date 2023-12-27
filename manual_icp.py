import cv2
import numpy as np
import open3d as o3d
from pyk4a import PyK4A

# 初始化Kinect
k4a = PyK4A()
k4a.start()

# 保存图像的计数器
image_count = 0
while True:
    # Get the capture
    capture = k4a.get_capture()
    if capture.depth is not None:
        # Get the point cloud
        point_cloud = capture.depth_point_cloud

        # Optional: Display the depth image
        depth_frame = capture.depth
        depth_image = cv2.convertScaleAbs(depth_frame, alpha=0.05)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Depth Image', depth_image)

        # Check user input
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Save the point cloud data
            filename = f"depth_image_{image_count}.ply"
            
            # Convert the point cloud to Open3D format
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(depth_frame)

            # Save point cloud in PLY format
            o3d.io.write_point_cloud(filename, point_cloud_o3d)

            image_count += 1
        elif key == 27:  # Press 'Esc' to exit
            break

# Stop Kinect
k4a.stop()


