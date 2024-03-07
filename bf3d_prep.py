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
from scipy.sparse import lil_matrix
import gc

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
        rxyz = np.array(pcd_mic.points)[:, np.newaxis, :] - np.array(centers_pcd.points)[np.newaxis, :, :]
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

        return r, theta_far, rxyz
    
    def calculate_similarity_and_find_edges(self, normals, indices, threshold=0.95, chunk_size=1000):
        n = normals.shape[0]
        edge_points = lil_matrix((n, 1), dtype=bool)
        # use the lil_matrix to store the sparse matrix
        similar_normals_sparse = lil_matrix((n, n), dtype=bool)

        for start_row in range(0, n, chunk_size):
            end_row = min(start_row + chunk_size, n)
            chunk_normals = normals[start_row:end_row]
            dot_products = np.dot(chunk_normals, normals.T)
            similarity = dot_products >= threshold
            print("calculating progress: ", end_row, " / ", n)

            for i, row in enumerate(range(start_row, end_row)):
                neighbor_indices = indices[row, 1:]
                is_edge = np.any(~similarity[i, neighbor_indices])
                edge_points[row] = is_edge

                # store in the sparse matrix
                similar_normals_indices = np.where(similarity[i])[0]
                similar_normals_sparse[row, similar_normals_indices] = True

            del dot_products, similarity, similar_normals_indices, neighbor_indices, is_edge, chunk_normals

        return edge_points, similar_normals_sparse

    def find_edge_points(self, similarity_matrix, indices):
        # Identify edge points as those having at least one neighbor with a dissimilar normal
        edge_points = np.any(~similarity_matrix[indices[:, 0], indices[:, 1:]], axis=1)
        return edge_points

    def build_tree_and_find_neighbors(self, points, k=4):
        from scipy.spatial import cKDTree
        # Build a k-d tree for efficient nearest neighbors search
        tree = cKDTree(points)
        # Find k nearest neighbors for each point
        distances, indices = tree.query(points, k=k+1)
        return tree, distances, indices

    def is_edge_point(self, point_index, indices, normals, normal_similarity_threshold=0.95):
        target_normal = normals[point_index]
        for neighbor_index in indices:
            if np.dot(normals[neighbor_index], target_normal) < normal_similarity_threshold:
                return True
        return False
    
    def calculate_gradients(self, points, values, edge_points, tree_info, similar_normals_sparse):
        tree, distances, indices = tree_info
        gradients = np.zeros((len(points), 3))  # Allocate array for gradient vectors

        # Step 1: Calculate gradients normally for all points
        for i in range(len(points)):
            for j in range(1, distances.shape[1]):  # Skip the first index since it's the point itself
                if distances[i][j] == 0:
                    continue  # Avoid division by zero
                direction = (points[indices[i][j]] - points[i]) / distances[i][j]
                gradient = (values[indices[i][j]] - values[i]) / distances[i][j]
                gradients[i] += gradient * direction
            gradients[i] /= (distances.shape[1] - 1)  # Normalize by the number of neighbors

        # Step 2: Adjust gradients for edge points based on similar normals
        for i in range(len(points)):
            if edge_points[i]:
                similar_normals_indices = similar_normals_sparse.rows[i]
                if similar_normals_indices:
                    average_gradient = np.mean(gradients[similar_normals_indices], axis=0)
                    gradients[i] += average_gradient

        return gradients


    def precompute_edge_points(self, points, edge_points,normals, threshold=0.95):
        dot_products = np.dot(normals, normals.T)
        similar_normals_indices = []
        for i in range(len(points)):
            if edge_points[i]:
                # Compute dot products of normals for this edge point with all others
                dot_products = np.dot(normals, normals[i])
                similar_normals_indices[i] = np.where(dot_products >= threshold)[0]
            else:
                similar_normals_indices[i] = []

        return similar_normals_indices


    def apply_eq(self, signal, fs, filter_freqs, filter_gains, nperseg=1024):
        """
        Apply a custom filter to a signal.

        Parameters:
        - signal: The original signal.
        - fs: The sampling frequency of the signal.
        - filter_freqs: The frequencies of the filter's frequency response.
        - filter_gains: The gains corresponding to the filter_freqs.

        Returns:
        - reconstructed_signal: The filtered signal.
        """

        from scipy.signal import stft, istft, windows, freqz, hann
        from scipy.interpolate import interp1d
        import matplotlib.pyplot as plt

        # Interpolate filter gains
        freqs_interp = np.linspace(0, fs/2, fs//2)
        interp_func = interp1d(filter_freqs, filter_gains, kind='linear', fill_value="extrapolate")
        interp_gains = interp_func(freqs_interp)

        # STFT parameters
        noverlap = nperseg // 2
        window = windows.hann(nperseg)

        # Perform STFT
        f, t, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary='zeros')

        # Apply filter in frequency domain
        Zxx_filtered = np.zeros_like(Zxx)
        for i, freq in enumerate(f):
            if freq <= fs/2:
                idx = np.argmin(np.abs(freqs_interp - freq))
                Zxx_filtered[i, :] = Zxx[i, :] * interp_gains[idx]
            else:
                # Mirror the filter gains for frequencies above Nyquist frequency
                idx = np.argmin(np.abs(freqs_interp - (fs - freq)))
                Zxx_filtered[i, :] = Zxx[i, :] * interp_gains[idx]

        # Inverse STFT to reconstruct the signal
        _, reconstructed_signal = istft(Zxx_filtered, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=True)

        return reconstructed_signal
    
    def apply_eq_fir(self, numtaps, frequencies, filter_rec, original_signal, fs):
        """derive a FIR filter from the frequency response and apply it to the signal."""
        from scipy.signal import firwin2, lfilter

        # make sure frequncies stoped with fs/2
        #fq = np.concatenate((frequencies, [fs/2]))
        #fr = np.concatenate((filter_rec, [filter_rec[-1]]))

        fir_filt = firwin2(numtaps, frequencies, filter_rec, fs = fs, antisymmetric=False)
        # Apply the filter to the signal
        filtered_signal = lfilter(fir_filt, 1.0, original_signal)

        return filtered_signal, fir_filt
    



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
        self.rxyz = None
        self.normals = None
        self.k = None

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

    def far_field_pressure(self, f, r, grad_p, theta, S, p, dpdn, dgdn):
        
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
        rho = 1.225 # air density
        k = self.k

        a = np.sqrt(S/np.pi)[:, np.newaxis]
        term = 2 * j1(k * a * np.sin(theta)) / (k * a * np.sin(theta))
        # p_hat = 2*I/(r) * S[:, np.newaxis] *np.exp(1j * ( - k * r)) * term
        # p_hat = 1j * omega * rho * u * S[:, np.newaxis] / (2 *np.pi *r) *np.exp(1j * ( - k * r)) * term
        # 1j * omega * rho * u =  - grad_p
        # u = p/(1j * omega *rho) + grad_p[:, np.newaxis]
        # u = -1/(1j * omega * rho) * (-grad_p[:, np.newaxis]) + 1j * p
        G = self.G(r, k)
        dpdn = dpdn[:, np.newaxis]

        self.datas = {"grad_p": grad_p,
                "r": r, 
                "theta": theta,
                    "S": S,
                    "a": a,
                    "term": term,
                    "dgdn": dgdn,
                    "p": p,
                    "f": f,
                    "G": G,
                    "dpdn": dpdn}
        

        p_hat = - G * dpdn * S[:, np.newaxis] + p * dgdn * S[:, np.newaxis]
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
        
        # p_hat =  1j * omega * rho * u / (2 *np.pi *r) *np.exp(1j * ( - k * r)) *S[:, np.newaxis]

        return (p_hat)

    def G(self, R, k):
        return np.exp(-1j * k * R) / R

    def grad_G(self, x, y, z, k, R):
        # Compute partial derivatives
        dG_dR = (-np.exp(-1j * k * R) / R**2) - (1j * k * np.exp(-1j * k * R) / R)
        grad_x = dG_dR * (x / R)
        grad_y = dG_dR * (y / R)
        grad_z = dG_dR * (z / R)
        
        return grad_x, grad_y, grad_z

    def dG_dn(self, r):
        x = self.rxyz[:,:,0].T
        y = self.rxyz[:,:,1].T
        z = self.rxyz[:,:,2].T
        n = self.normals

        grad_x, grad_y, grad_z = self.grad_G(x, y, z, self.k, r)
        return grad_x * n[:,0,np.newaxis] + grad_y * n[:,1,np.newaxis] + grad_z * n[:,2,np.newaxis]


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


    def pressure_at_array_point(self, r, theta_far, f_values, I_list, grad_p):
        S = self.compute_triangle_areas(self.mesh)

        # Compute Pf for all t values
        Pf_values = []
        for f, I, gp in zip(f_values, I_list, grad_p):
            # U[theta <= 0] = 0
            # display(np.count_nonzero(U))
            # Step Two, given the mesh volume velocity, get the total sound field.
            C = 343
            omega = 2 * np.pi * f
            self.k = omega / C
            dgdn = self.dG_dn(r)
            dpdn  = np.einsum('ij,ij->i',gp, self.normals)
            
            Pf_f = np.sum(self.far_field_pressure(f, r,
                                                  gp, theta_far, 
                                                  S, I[:,np.newaxis], 
                                                  dpdn, dgdn, 
                                                  ), axis=0)
            Pf_values.append(Pf_f)

        Pf_values = np.array(Pf_values)
        return Pf_values
    
    def pressure_at_cir_point(self, r, theta_far, f_values, I_list, grad_p):
        S = self.compute_triangle_areas(self.mesh)

        # Compute Pf for all t values
        Pf_values = []
        for f, I, gp in zip(f_values, I_list, grad_p):
            # U[theta <= 0] = 0
            # display(np.count_nonzero(U))
            # Step Two, given the mesh volume velocity, get the total sound field.
            C = 343
            omega = 2 * np.pi * f
            self.k = omega / C
            dgdn = self.dG_dn(r)
            dpdn  = np.sum(gp *self.normals, axis=1)
            
            Pf_f = self.far_field_pressure(f, r,
                                            gp, theta_far, 
                                            S, I[:,np.newaxis], 
                                            dpdn, dgdn, 
                                            )
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
