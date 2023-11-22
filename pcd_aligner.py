import open3d as o3d
import numpy as np
import datetime

class PointCloud_PreProcessor:
    def __init__(self, file_path):
        ply_file_path = file_path + 'kf_output.ply'
        self.file_path = ply_file_path
        self.pcd = None
        self.cropped_pcd = None

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

    def pcd_write(self):
        # Saving the mesh into current path, naming by time
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the date and time
        datetime_str = current_datetime.strftime("%Y%m%d_%H%M")

        # Ask for notes
        name_note = str(input("Any note on file name: "))

        # Save the point cloud
        filepath = f"PointCloud/[{name_note}]saved_pointcloud_{datetime_str}.ply"  # my file path
        o3d.io.write_point_cloud(filepath, self.cropped_pcd)





class PointCloud_PostProcessor:
    def reader():
        # get all files, also get the point cloud file for later use
        files = glob.glob("Mesh/*saved_mesh_*.ply")
        pointCloudFile = glob.glob("PointCloud/*saved_pointcloud_*.ply")

        # sorting files
        files.sort(key=os.path.getmtime)
        pointCloudFile.sort(key=os.path.getmtime)

        # print menu
        for i, file in enumerate(pointCloudFile):
            print(f"{i}: {pointCloudFile[i]}")

        # get user input
        selection = int(input("Enter the number of the file you want to view: "))

        # get file
        file_to_view = pointCloudFile[selection]
        # pcd_file_to_view = pointCloudFile[selection]

        # read file
        pcd = o3d.io.read_point_cloud(file_to_view)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5)

        # visualize using vertex normals method
        o3d.visualization.draw_geometries([mesh_frame, pcd])
        return pcd