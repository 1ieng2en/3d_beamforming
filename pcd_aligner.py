import open3d as o3d
import numpy as np

class PointCloudProcessor:
    def __init__(self, file_path):
        ply_file_path = file_path + 'kf_output.ply'
        self.file_path = ply_file_path
        self.pcd = None

    def pcd_read(self):
        # Load the .ply file
        self.pcd = o3d.io.read_point_cloud(self.file_path)
        return self.pcd

    def pcd_crop(self):
        """
        Allows user to pick points from the point cloud and returns the cropped point cloud.
        """
        print("For point selecting")
        print("1) Please pick points using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        
        print("For crop the model")
        print("Press Y to switch to the ortho view, \n K to lock the view and pick cropping area \n C to crop \n F to free view")
        print("\n After picking points, press Q to quit, cropped model will be saved in variable \"pcd\"")
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(self.pcd)
        vis.run()  # user picks points
        vis.destroy_window()

        # Get the picked points and the cropped geometry
        picked_points = vis.get_picked_points()
        cropped_geometry = vis.get_cropped_geometry()

        # Print coordinates of the picked points
        for i, idx in enumerate(picked_points):
            print(f"Point #{i + 1}: Index({idx}) - Coordinates {np.asarray(self.pcd.points)[idx]}")

        return cropped_geometry

