import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import open3d as o3d
import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import pandas as pd
import time
import copy
import scipy.spatial
import tables
import xml.etree.ElementTree as ET
from xml.dom import minidom

class bf3d_data_prep:
    def __init__(self, rootfolder, filetype, return_type = 'path'):
        self.rootfolder = rootfolder
        self.filetype = filetype
        self.current_path = self.rootfolder
        self.display_options(self.current_path)
        self.selected_file = None
        self.return_type = return_type
        self.rsf = None
        self.datas = None

    def display_options(self, folder_path):
        clear_output(wait=True)  # Clear the previous widgets
        options = self.get_options(folder_path)
        if not options:
            print("No folders or files matching the criteria.")
            return

        dropdown = widgets.Dropdown(
            options=options,
            description='Select:',
            disabled=False,
        )

        button = widgets.Button(description="Select")
        output = widgets.Output()

        display(dropdown, button, output)

        def on_button_clicked(b):
            with output:
                #if the files in the folder is end with the file type, then return the folder path
                if self.return_type == 'path':
                    if dropdown.value.endswith(self.filetype):
                        self.selected_file = os.path.join(folder_path, dropdown.value)
                        print(f"File selected: {self.selected_file}")
                    else:
                        self.current_path = os.path.join(folder_path, dropdown.value)
                        self.display_options(self.current_path)

        button.on_click(on_button_clicked)

    def get_options(self, folder_path):
        options = ['..']  # Option to go up one directory
        try:
            for item in os.listdir(folder_path):
                full_path = os.path.join(folder_path, item)
                if os.path.isdir(full_path) and not item.startswith('.'):
                    options.append(item)
                elif item.endswith(self.filetype) and os.path.isfile(full_path):
                    options.append(item)
        except Exception as e:
            print(f"Error accessing the directory: {e}")
            options = []

        return options
    
    def return_folder(self):
        """automatic scanning the files in the folder
        if the files in the folder is end with the file type, then return the folder path"""
        

    def convert_to_h5(self, recording, T = 10, ):
        """convert the file to h5 file"""
        tables.file._open_files.close_all()
        fs = (recording.shape[1])/T
        datamat7 = np.array(recording.T, dtype='float32')

        h5filemat7 = tables.open_file("temp.h5", mode='w', 
                                    title='three_sources')
        earraymat7 = h5filemat7.create_earray('/', 'time_data', obj=datamat7)
        display(earraymat7)
        h5filemat7.root.time_data.set_attr('sample_freq',fs)
        h5filemat7.close()

    def save_pcd_to_xml(self, pcd = None, xml_filename = "pcd_temp.xml", subgrid_name="default"):
        """
        Save Open3D point cloud coordinates to an XML file.

        :param pcd: Open3D point cloud object
        :param xml_filename: Path to the output XML file
        :param subgrid_name: Name of the subgrid (default is 'default')
        """
        # Extract points from the point cloud

        points = np.asarray(pcd.points)

        # Create XML structure
        root = ET.Element('root')
        for point in points:
            pos_element = ET.SubElement(root, 'pos', {
                'x': str(point[0]),
                'y': str(point[1]),
                'z': str(point[2]),
                'subgrid': subgrid_name
            })
        # Beautify and write to XML file
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open(xml_filename, "w") as xml_file:
            xml_file.write(xml_str)
            xml_file.close()
        
    
    def mesh_and_center(self, pcd_mic, cpcd):
        """
        this function will calculate the reverse sound field of the measured sound field
        input will take point cloud data and sound field data
        and the point cloud data will be used to calculate the reverse sound field pressure to the
        microphone array position
        """

        self.rsf = reverse_sound_field(centers_pcd=None, 
                                  mic_point=np.asarray(pcd_mic.points), 
                                  filtered_pcd=cpcd, 
                                  pressure_at_mesh=None, 
                                  frequency_list=None)

        cpcd_mesh = self.rsf.gen_mesh(pcd = cpcd)

        triangles = np.asarray(cpcd_mesh.triangles)
        vertices = np.asarray(cpcd_mesh.vertices)
        normals = np.asarray(cpcd_mesh.triangle_normals)
        centers = []

        for tri, normal in zip(triangles, normals):
            triangle_vertice = vertices[tri]
            center = np.mean(triangle_vertice, axis=0)
            centers.append(center)

        # load centers into o3d point cloud
        centers = np.array(centers)
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(centers)
        
        return cpcd_mesh, centers_pcd
    
    def cal_distance_and_angle(self, cpcd_mesh, centers_pcd, pcd_mic):
        """
        this function will calculate the distance and angle between the microphone array and the mesh centers
        """
        # calculate the distance between the mesh centers and the microphone array
        r = scipy.spatial.distance.cdist(np.array(pcd_mic.points), np.array(centers_pcd.points), metric="euclidean")
        r = r.T
        # calculate the angle between the mesh centers and the microphone array
        if self.rsf is None:
            self.rsf = reverse_sound_field(centers_pcd=None, 
                                  mic_point=np.asarray(pcd_mic.points), 
                                  filtered_pcd=cpcd_mesh, 
                                  pressure_at_mesh=None, 
                                  frequency_list=None)
            
        mic_to_center_vector = np.array(pcd_mic.points)[np.newaxis,:,:] - np.array(centers_pcd.points)[:,np.newaxis,:]
        r_alt = np.linalg.norm(mic_to_center_vector, axis=2)
        mic_to_center_vector = mic_to_center_vector / r_alt[:,:, np.newaxis]

        # calculate the mesh normal
        normals = np.asarray(cpcd_mesh.triangle_normals)
        theta_far = []

        for normal, mcv in zip(normals, mic_to_center_vector):
            angle = np.arccos(np.dot(normal, mcv.T))
            theta_far.append(angle)
        theta_far = np.array(theta_far)

        self.datas = {"r": r,
                    "theta_far": theta_far, 
                    "r_alt": r_alt,
                    "mcv": mic_to_center_vector, 
                    "normals": normals}

        return r, theta_far

    
    def build_tree_and_find_neighbors(self, points, k):
        from scipy.spatial import cKDTree

        # Build a k-d tree for efficient nearest neighbors search
        tree = cKDTree(points)

        # Precompute nearest neighbors for each point
        neighbors_info = [tree.query(point, k=k) for point in points]

        # Exclude the point itself from its list of neighbors
        neighbors_info = [(distances[1:], indices[1:]) for distances, indices in neighbors_info]

        return tree, neighbors_info
    
    def calculate_gradients_with_neighbors(self, points, values, neighbors_info):

        # Allocate array for gradients
        gradients = np.zeros_like(points[:,1])

        # Iterate over all points to calculate gradients
        for i, (distances, indices) in enumerate(neighbors_info):
            # Scalar value differences
            value_differences = values[indices] - values[i]

            # Avoid division by zero
            distances[distances == 0] = np.inf

            # Calculate weighted sum of differences
            weighted_diff = np.sum(value_differences / distances)

            # Assign calculated gradient
            gradients[i] = weighted_diff

        return gradients


class reverse_sound_field:
    """
    this class will calcualte the reverse sound field of the measured sound field
    input will take point cloud data and sound field data
    and the point cloud data will be used to calculate the reverse sound field pressure to the
    microphone array position
    """

    def __init__(self, centers_pcd, mic_point, filtered_pcd, pressure_at_mesh, frequency_list):
        self.mic = mic_point
        self.pcd = filtered_pcd
        self.pm = pressure_at_mesh
        self.frequency_range = frequency_list
        self.mesh = self.gen_mesh(filtered_pcd)
        self.datas = None
        self.centers_pcd = centers_pcd

    def gen_mesh(self, pcd):
        
        # variables for ball size, [lowest dimention, highest dimention, average dimention(starting point), increment, unit: m]

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        radii = [0.0005, 0.006, 0.001, 0.0005]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        rec_mesh.compute_vertex_normals()
        rec_mesh.compute_triangle_normals()
        return rec_mesh


    def volume_velocity(self, f, r, theta, S):
        """
        Calculate the volume velocity at the element
        
        Parameters:
            omega: Angular frequency
            t: Time
            k: Wave number
            r: distance from the point source to the element
            theta: Angle between the direction of propagation of the incident wave and the normal to the element surface
            S: Area of the element
        
        Returns:
            U: Volume velocity at the element
        """
        C = 343
        k = 2 * np.pi * f / C

        U = self.pm * (1/r + k) * S
        return U

    def far_field_pressure(self, f, r, grad_p, theta, S, p):
        
        """
        Calculate the far field approximation using the radiation of piston.
        
        Parameters:
            omega: Angular frequency
            t: Time
            k: Wave number
            rho: Air density
            Q: Volume velocity of the element
            a: Radius of the piston, assume to be sqrt(S/pi)
            theta: Angle between the direction of propagation of the incident wave and the normal to the element surface
        
        Returns:
            p_hat: Far field pressure due to the piston radiation, approximated
        """
        C = 343
        omega = 2 * np.pi * f
        k = omega / C
        rho = 1.225 # air density

        a = np.sqrt(S/np.pi)[:, np.newaxis]
        term = 2 * j1(k * a * np.sin(theta)) / (k * a * np.sin(theta))
        # p_hat = 2*I/(r) * S[:, np.newaxis] *np.exp(1j * ( - k * r)) * term
        # p_hat = 1j * omega * rho * u * S[:, np.newaxis] / (2 *np.pi *r) *np.exp(1j * ( - k * r)) * term
        # 1j * omega * rho * u =  - grad_p

        # u = p/(1j * omega *rho) + grad_p[:, np.newaxis]
        u = -1/(1j * omega * rho) * (-grad_p[:, np.newaxis]) + 1j * p

        self.datas = {"grad_p": grad_p,
                "r": r, 
                "theta": theta,
                    "S": S,
                    "a": a,
                    "term": term,
                    "k": k,
                    "p": p,
                    "f": f,
                    "u": u}

        # p_hat =  - grad_p[:, np.newaxis]* S[:, np.newaxis] / (2 *np.pi *r) *np.exp(1j * ( - k * r)) * term
        # p_hat =  - grad_p[:, np.newaxis]/ (2 *np.pi *r) *np.exp(1j * ( - k * r)) * term
        # is it really make sense to multiply the S? becouse S is the 
        # area of the element, and the area is for calculate the volume velocity
        # but then the volume velocity will be biased?
        
        # p_hat = 2*I/(r)  *np.exp(1j * ( - k * r)) * term
        
        # for explore: what if I don't include term as well?
        # p_hat = 2*I/(r)  *np.exp(1j * ( - k * r))

        # Assume it's a piston on the baffle is not correct, but we should use rayleigh integral
        # p_hat = j omega rho u/(2 pi r) exp(-jkr) dxdydz (integral over the surface)
        # p_hat = j omega rho u/(2 pi r) exp(-jkr) dS (integral over the surface)
        
        p_hat =  1j * omega * rho * u / (2 *np.pi *r) *np.exp(1j * ( - k * r)) *S[:, np.newaxis] * term

        return (p_hat)

    def triangle_area(self, vertices):
        """Compute the area of a triangle given its vertices."""
        a = np.linalg.norm(vertices[0] - vertices[1])
        b = np.linalg.norm(vertices[1] - vertices[2])
        c = np.linalg.norm(vertices[2] - vertices[0])
        s = 0.5 * (a + b + c)
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def compute_triangle_areas(self, mesh):
        """Compute areas for all triangles in the mesh."""
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        return np.array([self.triangle_area(vertices[tri]) for tri in triangles])

    def angle_between_vectors(self, vector, normal):
        """Compute the angle in degrees between a given vector and a normal."""
        dot_product = np.dot(vector, normal)
        magnitudes = np.linalg.norm(vector) * np.linalg.norm(normal)
        return np.degrees(np.arccos(dot_product / magnitudes)), magnitudes

    def vector_from_point_to_triangle_center(self, point, triangle_vertices, axis_in):
        """Compute the vector from a given point to the center of the triangle."""
        center = np.mean(triangle_vertices, axis = axis_in)
        return center - point
    
    def monopole_Pressure(self, R, f): # define the monopole pressure in relate with distance, time, and wavenumber
        # Your code here
        c = 343
        k = 2 * np.pi * f / c
        Pm = 1/R * np.exp(1j * ( - k * R))
        return Pm

    def compute_angles_from_point_to_normals(self, mesh, point):
        """Compute angles between the vector from a point to triangle centers and all triangle normals in the mesh."""
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.triangle_normals)
        angles = []
        mags = []
        vector_to_centers = []

        for tri, normal in zip(triangles, normals):
            triangle_vertice = vertices[tri]
            vector_to_center = self.vector_from_point_to_triangle_center(point, triangle_vertice, 0)
            angle, mag = self.angle_between_vectors(vector_to_center, normal)
            angles.append(angle)
            mags.append(mag)
            vector_to_centers.append(vector_to_center)
        
        return np.array(angles), np.array(mags)

    def pressure_at_array_point(self, r, theta_far, f_values, I_list, grad_p):
        S = self.compute_triangle_areas(self.mesh)

        # Compute Pf for all t values
        Pf_values = []
        for f, I, gp in zip(f_values, I_list, grad_p):
            # U[theta <= 0] = 0
            # display(np.count_nonzero(U))
            # Step Two, given the mesh volume velocity, get the total sound field.
            Pf_f = np.sum(self.far_field_pressure(f, r,gp, theta_far, S, I[:,np.newaxis]), axis=0)
            Pf_values.append(Pf_f)

        Pf_values = np.array(Pf_values)
        return Pf_values

    def calculate_reverse_sound_field(self):
        """
        calculate the reverse sound field pressure to the microphone array position
        """
        pass

    def get_significant_eva(self, eva):
        """
        Get the significant eigenvalues of the Cross Spectral Matrix.

        Parameters:
        eva (np.ndarray): Eigenvalues of the Cross Spectral Matrix.

        Returns:
        np.ndarray: Significant eigenvalues.
        """
        # Calculate the mean and standard deviation of the eigenvalues
        mean = np.mean(eva)
        std = np.std(eva)

        # Set a threshold value as a multiple of the standard deviation
        threshold = 2  # Adjust this value as per your requirement

        # Filter out eigenvalues that are significantly larger than the mean
        significant_eva = eva[eva > mean + threshold * std]

        count = len(significant_eva)

        return significant_eva


    def run(self):
        self.calculate_reverse_sound_field()
        return self.reverse_sound_field
