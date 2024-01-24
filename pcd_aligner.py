import glob
import os
import open3d as o3d
import numpy as np
import datetime
import clipboard

import io
import re
import contextlib

import xml.etree.ElementTree as ET


class PointCloud_PreProcessor:
    def __init__(self, file_path):
        self.ply_file_path = file_path
        self.pcd = None
        self.vis = None
        self.cpcd = None
        self.pcd_mic = None
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    def pcd_read(self):
        file_path = self.ply_file_path + 'kf_output.ply'
        self.pcd = o3d.io.read_point_cloud(file_path)
        return self.pcd

    def pcd_load(self, filetype = ".ply"):
        # get file
        file_to_view = self.read_and_choose(filetype=filetype)

        # read file
        self.pcd = o3d.io.read_point_cloud(file_to_view)
        return self.pcd
    
    def read_and_choose(self, filetype = ".ply"):
        # get all files, also get the point cloud file for later use
        file_to_view = glob.glob(f"{self.ply_file_path}/*{filetype}")
        # print(file_to_view, self.ply_file_path)

        # sorting files
        file_to_view.sort(key=os.path.getmtime)

        # print menu
        for i, file in enumerate(file_to_view):
            print(f"{i}: {file_to_view[i]}")

        # get user input
        selection = int(input("Enter the number of the file you want to view: "))

        # get file
        file_to_view = file_to_view[selection]
        return file_to_view


    def pcd_crop(self, pcd = None, title = "Point Cloud Cropping", save = True, mesh_frame = None):
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
        
        if pcd is None:
            pcd = self.pcd
        if pcd == "cpcd":
            pcd = self.cpcd
        vis = o3d.visualization.VisualizerWithEditing()

        vis.create_window(window_name = title)
        if mesh_frame is not None:
            mesh_frame = np.array(self.mesh_frame.vertices)
            mesh_frame_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_frame))
            combined_points = np.concatenate((np.asarray(pcd.points), np.asarray(mesh_frame_pcd.points)), axis=0)
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

            vis.add_geometry(combined_pcd)
        else:
            vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()

        # Get the picked points and the cropped geometry
        if save:
            self.cpcd = vis.get_cropped_geometry()
        self.vis = vis

        # picked_points = vis.get_picked_points()
        # Print coordinates of the picked points
        #for i, idx in enumerate(picked_points):
        #    print(f"Point #{i + 1}: Index({idx}) - Coordinates {np.asarray(self.pcd.points)[idx]}")

        return vis.get_cropped_geometry()

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
        print("Point cloud saved.")

    def pcd_show(self, pcd = "cpcd"):
        if pcd is None:
            pcd = self.pcd
        if pcd == "cpcd":
            pcd = self.cpcd
        if pcd == "mic":
            pcd = self.pcd_mic
        o3d.visualization.draw_geometries([self.mesh_frame, pcd]) # visualize the point cloud
        return

##############################################################################################################
# Extract data from clipboard
# for pcd align

    def extract_clipboard_data(self):
        clipboard_data = clipboard.paste()
        # print(f"Clipboard data: {clipboard_data}")  # Add this line to debug
        lookat = [float(x) for x in re.findall(r'"lookat" : \[ (.*?) \]', clipboard_data)[0].split(',')]
        front = [float(x) for x in re.findall(r'"front" : \[ (.*?) \]', clipboard_data)[0].split(',')]
        up = [float(x) for x in re.findall(r'"up" : \[ (.*?) \]', clipboard_data)[0].split(',')]
        return lookat, front, up

    def compute_rotation_matrix(self,source_front, source_up, target_front, target_up):
        # Compute rotation matrix to align source_front to target_front
        v = np.cross(source_front, target_front)
        c = np.dot(source_front, target_front)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix_front = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        
        # Rotate source_up using rotation_matrix_front
        rotated_up = rotation_matrix_front @ np.array(source_up)
        
        # Compute rotation matrix to align rotated_up to target_up
        v = np.cross(rotated_up, target_up)
        c = np.dot(rotated_up, target_up)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix_up = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        
        # Combine the two rotation matrices
        final_rotation_matrix = rotation_matrix_up @ rotation_matrix_front
        
        return final_rotation_matrix

    def apply_transformations(self, pcd, translation_vector, rotation_matrix):
        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [-x for x in translation_vector]
        
        # Create a 4x4 rotation matrix
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix
        
        # Apply transformations
        pcd.transform(translation_matrix)
        pcd.transform(rotation_matrix_4x4)
        
        return pcd

    def apply_rotation(self, pcd, points):
        """ Apply rotation based on user-selected points """
        point_x = np.array(points[0])
        origin = np.array(points[1])
        point_y = np.array(points[2])

        # define new x-axis
        x_axis = point_x - origin
        x_axis /= np.linalg.norm(x_axis)

        # calculate temporary y-axis
        temp_y_axis = point_y - origin
        temp_y_axis /= np.linalg.norm(temp_y_axis)

        # calculate new z-axis
        z_axis = np.cross(x_axis, temp_y_axis)
        z_axis /= np.linalg.norm(z_axis)

        # recalculate y-axis
        y_axis = np.cross(z_axis, x_axis)

        # create rotation matrix
        R = np.vstack([x_axis, y_axis, z_axis])

        # apply rotation
        pcd.rotate(R, center=origin)

        return origin

    def apply_translation(self, pcd, origin):
        """ apply translation based on user-selected points """
        pcd.translate(-origin)
        return pcd
    
    def create_plane_mesh(self, z):
        """ create a mesh plane """
        # defin vertices
        vertices = np.array([[0, 0, z],  # origin
                            [1, 0, z],  # 1m in x direction
                            [1, -1, z],  # 1m in x and -1m in y direction
                            [0, -1, z]]) # 1m in -y direction

        # define triangles
        triangles = np.array([[0, 1, 2], 
                            [2, 3, 0]])

        # create mesh
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        plane_mesh.compute_vertex_normals()

        return plane_mesh
    def pick_points_and_get_coordinates(self):
        # create a string stream to hold the output
        str_io = io.StringIO()

        # redirect stdout to the string stream and perform picking
        with contextlib.redirect_stdout(str_io):
            self.pcd_crop(pcd = "cpcd", title="coordinate allignment - copy the view direction and pick coordinate points") # get picked points

        # get the output
        output = str_io.getvalue()
        print(output)
        str_io.close()

        # extract coordinates from the output
        coordinates = re.findall(r"\d+ \((.*?)\)", output)
        points = [tuple(map(float, coord.split(', '))) for coord in coordinates]

        return points
        

##############################################################################################################
# Align point cloud coordinates
    
    def rotate_to_view(self):
        
        # Extract data from clipboard
        lookat, front, up = self.extract_clipboard_data()

        # Define target front and up vectors
        target_front = [0.0, 0.0, 1.0]
        target_up = [0.0, 1.0, 0.0]

        # Compute rotation matrix
        rotation_matrix = self.compute_rotation_matrix(front, up, target_front, target_up)

        # Apply transformations and visualize the result
        self.cpcd = self.apply_transformations(self.cpcd, lookat, rotation_matrix)
        
    def coordinates_align(self):
        """ Align point cloud coordinates and regulize the point cloud default view direction"""

        # ask user to pick points
        picked_points = self.pick_points_and_get_coordinates()
        print(f"Picked points: {picked_points}")

        if len(picked_points) == 3:
            # create plane mesh
            # plane_mesh = self.create_plane_mesh(0)
            # plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # grey
            # apply rotation
            origin = self.apply_rotation(self.cpcd, picked_points)
            # apply translation
            self.cpcd = self.apply_translation(self.cpcd, origin)
            o3d.visualization.draw_geometries([self.cpcd, self.mesh_frame])

        else:
            print("need 3 points to align")
        return self.cpcd, picked_points
    
##############################################################################################################
# align array with point cloud
    def mic_array(self, filename, x,y,z):
        # import the microphone positions
        tree = ET.parse(filename)
        root = tree.getroot()

        values = [(int(item.attrib["Name"][6:]), item.attrib['x'], item.attrib['y'],item.attrib['z']) for item in root.findall('pos')]
        values = np.array(values, dtype=float)
        values[:,1] += x
        values[:,2] += y
        values[:,3] += z
        return values
    
    def load_mic_array(self, filename = 'P36D45_f14.xml',x = 0 ,y = 0 ,z = 0):
        # import the microphone positions
        mics_number = self.mic_array(filename,x,y,z)
        mics = mics_number[:,1:]

        self.pcd_mic = o3d.geometry.PointCloud()
        self.pcd_mic.points = o3d.utility.Vector3dVector(mics)

##############################################################################################################

    def array_align(self):
        source = self.pcd_mic
        target = self.cpcd

        self.pcd_crop(pcd = target, title="choose 3 points on the pcd", mesh_frame= None)
        picked_id_target = self.vis.get_picked_points()
        self.pcd_crop(pcd = source, title = "choose 3 corrsponding points on the array", save = False, mesh_frame= True)
        picked_id_source = self.vis.get_picked_points()

        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

        # point-to-point ICP for refinement
        print("Perform point-to-point ICP refinement")
        # threshold = 0.03  # 3cm distance threshold
        #reg_p2p = o3d.pipelines.registration.registration_icp(
        #    source, target, threshold, trans_init,
        #    o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # transformation = reg_p2p.transformation

        #source.transform(transformation)
        source.transform(trans_init)
        spheres = []

        self.pcd_mic = source
        self.cpcd = target
        return

    def paint_uniform_color(self, pcd = None, color = [0.5, 0.5, 0.5]):
        if pcd is None:
            pcd = self.cpcd
        pcd.paint_uniform_color(color)
        return pcd

    def pcd_show_highlight(self):

        source = self.pcd_mic
        target = self.cpcd

        spheres = []

        # create a sphere for each point in the point cloud
        for i, point in enumerate(np.asarray(source.points)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # set radius
            sphere.translate(point)
                # if this is the first point, color it red, otherwise, color it white
                # by tis way it is easier to identify the first point to avoid miss alignment.
            if i == 0 or i == 1:
                colors = np.array([[1, 0, 0] for _ in range(len(sphere.vertices))])  # red
                sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            spheres.append(sphere)

        o3d.visualization.draw_geometries(spheres + [target])  # draw all spheres and target point cloud

        # o3d.visualization.draw_geometries([mesh, target])
        return
    
    def remove_array(self):
        self.pcd_crop(pcd="cpcd", title="remove array")
        return self.cpcd
    
    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return

        # save the point clouds
    def save_models(self, models, file_to_view):
        directory = f"{file_to_view}"
        self.ensure_dir(directory)  # ensure the directory exists
        for i, model in enumerate(models):
            filename = f"{directory}/model_{i}.ply"  # change the file name here
            try:
                o3d.io.write_point_cloud(filename, model)
                print(f"Model saved as {filename}")
            except Exception as e:
                print(f"Failed to save {filename}: {e}")

    def pcd_write_group(self):

        userinput = input("save this file as new? y/[n]")
        if userinput == "y":
            # get the current time
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save the first group
            choice = input("Enter the prefix:")
            # uncomment the following line if performed ray tracing...
            self.save_models([self.pcd_mic, self.pcd, self.cpcd], f"postPCD/{choice}_{current_time}")
        else:
            self.ply_file_path = 'postPCD'
            file_to_view = self.read_and_choose(filetype='')
            self.save_models([self.pcd_mic, self.pcd, self.cpcd], file_to_view)