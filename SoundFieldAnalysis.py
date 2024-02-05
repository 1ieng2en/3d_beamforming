import numpy as np
import scipy.spatial
from scipy.signal import welch
import pyvista as pv
import cvxpy as cp
import open3d as o3d
from numpy import ndarray
import pcd_aligner




class SoundFieldAnalysis:
    """
    This class performs sound field analysis.
    It allows for the calculation of sound pressure at various points in space and visualizes
    the sound field in 3D.

    Attributes:
    mic_array (numpy.ndarray): Array of microphone positions.
    sound_field (numpy.ndarray): Measured sound pressure data from microphones.
    points (numpy.ndarray): Points in space where sound pressure is calculated.
    freq (float): Frequency at which to perform the analysis.
    seg_size (int): Size of segments to split the point cloud for processing.
    nperseg (int): Number of data points per segment for FFT calculation.
    """

    def __init__(self, mic_pcd, sound_field, pcd, freq, seg_size=2000, nperseg=1600):
        self.mic_pcd = mic_pcd
        self.mic_array = np.array(self.mic_pcd.points)
        self.sound_field = sound_field
        self.pcd = pcd
        self.points = np.array(self.pcd.points)
        self.freq = freq
        self.seg_size = seg_size
        self.nperseg = nperseg
        self.S = 0
        self.Pxy = 0
        self.f = 0
        self.plotc = 0
        self.distance = 0
        self.index = 0
        self.k = 0
        self.eigVal = 0
        self.pv_mesh = None
        self.origional_mic_pcd = None


    def load_mic_array(self):
        """
        Load the microphone array.
        """
        cpcd_path = f"../data/pcd/pcd_mic.pcd"
        processor = pcd_aligner.PointCloud_PreProcessor(cpcd_path)
        processor.load_mic_array()
        self.origional_mic_pcd = np.array(processor.pcd_mic.points)


    def get_freq_list(self, freq_range, T = 10):
        """
        Get the list of frequencies available in the sound field data.
        """

        N = np.shape(self.sound_field)[1]
        fs = N / T
        f, Pxy = welch(self.sound_field, fs, noverlap=0, nperseg=self.nperseg, return_onesided=True, axis=-1)
        Pxy = fs / 1.28 * Pxy

        if freq_range is not None:
            f = f[freq_range[0] <= f]
            f = f[f <= freq_range[1]]
        
        self.Pxy = np.array(Pxy)
        self.f = f
        return f

    def calculate_sparse(self, freq_range = None, T = 10):
        """
        Calculate the sparse matrix.
        under the frequency range mode, you don't need to run this method in the loop, run the get_freq_list() method instead.
        """
        
        N = np.shape(self.sound_field)[1]
        fs = N / T
        f, Pxy = welch(self.sound_field, fs, noverlap=0, nperseg=self.nperseg, return_onesided=True, axis=-1)
        Pxy = fs / 1.28 * Pxy
        self.Pxy = np.array(Pxy)
        self.f = f
        self.load_mic_array()

        if freq_range is not None:
            f = f[freq_range[0] <= f]
            f = f[f <= freq_range[1]]
            return f


    def calculate_csm(self):
        """
        Select the frequency at which to perform the analysis.
        """
        m, _ = self.sound_field.shape
        index = np.argmin(np.abs(self.f - self.freq))
        self.index = index

        C = 343
        omega = 2 * np.pi * self.f[index]
        self.k = omega / C

        Csm = np.zeros((m, m), dtype=complex)

        for i in range(m):
            for j in range(m):
                if i != j: # diangional removal
                    Csm[i, j] = np.sqrt(self.Pxy[i, index]) * np.sqrt(self.Pxy[j, index])
        self.S = Csm

    
    def find_transformed_origin(self, original_points, transformed_points):
        """
        Find the transformed coordinates of the origin given sets of original and transformed points.
        
        :param original_points: np.array of shape (n, 3) representing n original points.
        :param transformed_points: np.array of shape (n, 3) representing n transformed points.
        :return: The coordinates of the transformed origin point.
        """
        # Ensure the points are in correct shape
        if original_points.shape != transformed_points.shape or original_points.shape[1] != 3:
            raise ValueError("Both sets of points must have the same shape and be 3-dimensional.")

        # Calculate the centroids of the point sets
        centroid_original = np.mean(original_points, axis=0)
        centroid_transformed = np.mean(transformed_points, axis=0)

        # Center the points around the centroids
        centered_original = original_points - centroid_original
        centered_transformed = transformed_points - centroid_transformed

        # Compute the rotation matrix using SVD
        H = np.dot(centered_transformed.T, centered_original)
        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = np.dot(U, Vt)

        # Compute the translation vector
        translation_vector = centroid_transformed - np.dot(rotation_matrix, centroid_original)

        # Apply the transformation to the origin (0,0,0)
        transformed_origin = np.dot(rotation_matrix, [0, 0, 0]) + translation_vector

        return transformed_origin


    def DAS(self, points_subset):
        m, _ = self.sound_field.shape

        # calculate the transformed origin
        transformed_origin = self.find_transformed_origin(self.origional_mic_pcd[0:2, :], self.mic_array[0:2, :])
        # calculate the distance from the origin to the focus points
        r = scipy.spatial.distance.cdist([transformed_origin], points_subset, metric="euclidean")
        # calculate the distance from the mic array to individual microphone points
        rmic = scipy.spatial.distance.cdist([transformed_origin], self.mic_array, metric="euclidean")

        rmr = scipy.spatial.distance.cdist(self.mic_array, points_subset, metric="euclidean")
        vm = np.exp(-1j * self.k * rmri)/rmri

        rv = np.abs(rmic.T - r)

        Vmn = np.zeros((m, m, len(points_subset)), dtype=complex)
        Vmn2 = np.zeros((m, m, len(points_subset)), dtype=complex)

        for i in range(m):
            for j in range(m):
                if i != j: # diangional removal
                    #calculate the individual distance
                    rmri = rmr[i]
                    rmrj = rmr[j]
                    
                    # calculate steering vector
                    vm = np.exp(-1j * self.k * rmri)/rmri
                    vn = np.exp(-1j * self.k * rmrj)/rmrj
                    # calculate the sterring vector matrix
                    Vmn[i, j, :] = vm * vn.conj()
                    #Vmn2[i, j, :] = (np.real(vm) ** 2) * (np.real(vn) ** 2)
                    # Vmn2[i, j, :] = np.conj(vm) * vn
                    Vmn2[i, j, :] = (1/rmri)**2 * (1/rmrj)**2
                    # Vmn2[i, j, :] = (vm ** 2) * (vn ** 2)

        # calculate the sound pressure
        #h = (Vmn)
        #g = self.S[:,:,None]
        #J4 = np.sum(( h.swapaxes(1, 0).conj() * g )**2, axis=(0, 1)) / np.sum((h.swapaxes(1,0).conj() * h), axis=(0, 1))
        
        # J4 = ( h.swapaxes(1, 0).conj() * g )**2 / h.swapaxes(1,0).conj() @ h
        Jup = self.S[:, :, None] * (Vmn)

        #Vmn2_real = np.sqrt(np.real(Vmn2))
        #Vmn2_imag = np.sqrt(np.imag(Vmn2))
        result = 1 / np.sqrt(12*11) * (Jup.sum(axis=(0, 1))) / (np.sqrt(Vmn2.sum(axis=(0, 1))))

        #result = 1 / np.sqrt(36*35) * ((Jup.sum(axis=(0, 1))) / ((Vmn2_real+1j*Vmn2_imag).sum(axis=(0, 1))))
        # result = 1 / np.sqrt(36*35) * np.real(Jup.sum(axis=(0, 1)))
        # result = 1 / np.sqrt(36*35) * ((Jup.sum(axis=(0, 1))) / (Vmn2.sum(axis=(0, 1))))
        return result, self.f


    
    def MUSIC_gen_eig(self, eigVal, eigVec, signal_number):
        idx = []
        eigVal_copy = eigVal.copy()  # create a copy of eigVal

        for i in range(signal_number):
            idx_largest = np.argmax(eigVal_copy)
            eigVal_copy[idx_largest] = -np.inf  # note the largest value
            idx.append(idx_largest)

        eigVal = np.delete(eigVal, np.array(idx))
        eigVec = np.delete(eigVec, np.array(idx), axis=1)
        
        return eigVal, eigVec

    def pseudo_distance(self, points_subset):
        '''
        Calculate the pseudo distance
        The distance is calculated by the distance to the mic array center - micarray center to individual mic
        '''
        m, _ = self.sound_field.shape
        distance = scipy.spatial.distance.cdist(self.mic_array, points_subset, metric="euclidean")

    

    def MUSIC(self, signal_number):
        S = self.S
        eigVal, eigVec = np.linalg.eig(S)
        eigVal, eigVec = self.MUSIC_gen_eig(eigVal, eigVec, signal_number)

        E_n = ((eigVec)*eigVal)
        omega = 2*np.pi*self.f[self.index]
        C = 343
        k = omega/C
        distance = scipy.spatial.distance.cdist(self.mic_array, self.points, metric="euclidean")
        v = np.exp(-1j * k * distance) / distance
        P = []
        for i in range(len(distance[2,:])):
            a = v[:,i]
            Pi =  1/((a.conj().T @ E_n @ E_n.conj().T @ a))**2
            P.append(Pi)
        return np.array(P)



    def CS(self,**kwargs):
        '''
        Compressive sensing method
        kwargs:
        initial_x: initial value of x
        eps: the value of eps

        '''

        omega = 2*np.pi*self.f[self.index]
        C = 343
        k = omega/C
        distance = scipy.spatial.distance.cdist(self.mic_array, self.points, metric="euclidean")
        v = np.exp(-1j * k * distance) / distance

        S = self.S
        eigVal, eigVec = np.linalg.eig(S)
        # eigVal = np.sqrt(eigVal)
        # b = np.sqrt(np.diag(S))
        #b = 1/(eigVec[:,1:]*np.sqrt(eigVal[1:]))
        # b = eigVec[:,0] * eigVal[0]
        b = np.sqrt(self.Pxy[:,self.index])

        # eps = cp.Variable()
        eps = kwargs.get("eps", 1e-5)
        x = cp.Variable(len(v[2,:]),  nonneg=True)

        # set the initial value of x
        x.value = kwargs.get("initial_x", np.ones(len(v[2,:]))*1e-6)

        # objective = cp.Minimize(cp.sum(x) + 10* eps)
        # constraints = [cp.norm(v @ x - b) <= eps]
        # Introduce a new variable for the maximum value of x
        # x_max = cp.Variable()

        objective = cp.Minimize(cp.sum(x))
        # constraints = [cp.sum_squares(v @ x - b)<=eps/10]
        # constraints = [cp.norm(v.T@(b - v @ x))<=2]
        #constraints = [cp.norm(b - v @ x)<=2, 
                        #x <= x_max,                         # Every element of x is less than or equal to x_max
                        #x_max >= 20e-7,                    # Your constraint on the maximum level of x
                        #]
        
        constraints = [cp.sum_squares(b - v @ x)<=eps]
        # constraints = [cp.norm(v.T@(b - v @ x))<=2]
        ##########################
        # N = v.shape[1]  # Number of columns in v
        # select = cp.Variable([N,1], boolean=True)  # Binary selection variable

        # Objective and other constraints
        # ...

        # Add selection constraints
        # constraints += [cp.sum(select) == 1]  # Only one column is selected

        # Formulate the constraint using the selected column
        # select_column = cp.reshape(select, (N, 1))

        # selected_column = v @ select  # This will result in a weighted sum of the columns, effectively selecting one.

        # selected_column = cp.sum(selected_column, axis=1)

        # constraints += [cp.norm(b - selected_column * x_max) <= 20e-4]

        # Define and solve the problem (use a MIP-compatible solver)
        prob = cp.Problem(objective, constraints)
        # result = prob.solve(solver=cp.ECOS_BB)  # Example of a MIP-compatible solver

###############################################


        #prob = cp.Problem(objective, constraints)

        max_iters = kwargs.get("max_iters", 6000)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver=cp.SCS, max_iters=max_iters, verbose=True, eps = 1e-5)
        # result = prob.solve(solver=cp.ECOS, verbose=True)
#####################################
        # if prob.status == "optimal":
        optimal_x = x.value
        print("eps:" ,eps)
        print("result s sum:", result)

        return optimal_x
    def CS2(self):
        
        omega = 2*np.pi*self.f[self.index]
        C = 343
        k = omega/C
        distance = scipy.spatial.distance.cdist(self.mic_array, self.points, metric="euclidean")
        v = np.exp(-1j * k * distance) / distance

        S = self.S
        eigVal, eigVec = np.linalg.eig(S)
        # eigVal = np.sqrt(eigVal)
        # b = np.sqrt(np.diag(S))
        #b = 1/(eigVec[:,1:]*np.sqrt(eigVal[1:]))
        b = eigVec[:,0] * eigVal[0]
                # eps = cp.Variable()
        eps = 1e-5
        x = cp.Variable(len(v[2,:]),  nonneg=True)
        # objective = cp.Minimize(cp.sum(x) + 10* eps)
        # constraints = [cp.sum_squares(v @ x - b) <= eps, eps >= 0]

        objective = cp.Minimize(cp.sum(x))
        # constraints = [cp.sum_squares(v @ x - b)<=eps/10]
        # constraints = [cp.norm(v.T@(b - v @ x))<=2]
        # constraints = [cp.norm(b - v @ x)<=2]
        constraints = [cp.norm(b - v @ x)<=0.002]
        # constraints = [cp.norm(v.T@(b - v @ x))<=2]
        prob = cp.Problem(objective, constraints)

        max_iters = 3000

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver=cp.ECOS, max_iters=max_iters, verbose=True)

        # if prob.status == "optimal":
        optimal_x = x.value
        print("eps:" ,eps)
        print("result s sum:", result)

        return optimal_x
    

    def deconv_beamforming(self, signal, kernel):
        """
        Deconvolution beamforming.
        """
        # Calculate the FFT of the signal and kernel
        signal_fft = np.fft.fft(signal)
        kernel_fft = np.fft.fft(kernel)

        # Perform deconvolution
        result_fft = signal_fft / kernel_fft

        # Calculate the inverse FFT
        result = np.fft.ifft(result_fft)
        
        return result


    def calculate_splice(self, mode = "DAS"):
        """
        Calculate sound pressure with spliced point cloud. For large point clouds, the calculation can improve the performance.
        """
        split_points = np.array_split(self.points, np.round(self.points.shape[0] / self.seg_size))
        J = []
        if mode == "DAS":
            for splitP in split_points:
                Js, _ = self.DAS(splitP)
                J.append(Js)
            result_J = np.concatenate(J, axis=0)
        return result_J
    
    
    
    def create_mesh_from_grid(self, points, rows, cols):
        # Initialize lists for triangles and vertices
        vertices = points
        triangles = []

        # Create two triangles for each grid cell
        for i in range(rows - 1):
            for j in range(cols - 1):
                idx = i * cols + j
                triangles.append([idx, idx + cols, idx + 1])
                triangles.append([idx + 1, idx + cols, idx + cols + 1])

        # Create the mesh
        pv_mesh = pv.PolyData(vertices, triangles)
        return pv_mesh
    
    def add_slicer(self, plotter, mode, p_range, center = [0,0], dynamic_range= 5, max_crop = 0, plane="xy", position=0, size=[1, 1], density=100, plot_mesh = True):
        # Create a meshgrid for the specified plane
        x = np.linspace(center[0]-size[0]/2, center[0]+size[0]/2, density)
        y = np.linspace(center[1]-size[1]/2, center[1]+size[1]/2, density)
        xx, yy = np.meshgrid(x, y)

        # Generate points based on the specified plane
        if plane == "xy":
            points = np.vstack((xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), position))).T
        elif plane == "yz":
            points = np.vstack((np.full_like(xx.flatten(), position), xx.flatten(), yy.flatten())).T
        elif plane == "xz":
            points = np.vstack((xx.flatten(), np.full_like(xx.flatten(), position), yy.flatten())).T
        else:
            print(f"Invalid plane: {plane}")
            return None
        
        temppoints = self.points # temporary store the points

        try:
            self.points = points # replace the points with the plane points, for calculation
            if plot_mesh:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pv_mesh = self.gen_mesh(pcd)
                cloud = pv.PolyData(pv_mesh)
            else:
                cloud = pv.PolyData(points)

            plotc, _ = self.gen_result(mode)
            if p_range is None:
                plotc_clamped, p_range = self.clamp(plotc, dynamic_range, max_crop)
            else:
                plotc_clamped = np.clip(plotc, p_range[0], p_range[1])
            cloud["Sound Pressure(dB)"] = plotc_clamped
            plotter.add_mesh(cloud, cmap='rainbow', scalars='Sound Pressure(dB)',show_scalar_bar = True,  opacity="linear")
        finally:
            self.points = temppoints # restore the points
        return plotter, cloud
    
    def gen_result(self, mode, **kwargs):
        '''
        generate the result of the mode
        kwargs:
        initial_x: initial value of x
        eps: the value of eps
        frequency range: the frequency range of the result'''

        if mode == "DAS":
            result_J = self.calculate_splice(mode = "DAS")
            plotc = 20 * np.log10(np.real(result_J / 20e-6))

        elif mode == "DAS-Non-splice":
            result_J, f = self.DAS(self.points)
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))

        elif mode.startswith("MUSIC"):
            result_J = self.MUSIC(signal_number=int(mode[-1]))
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))

        elif mode == "CS":
            result_J = self.CS(**kwargs)
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))

        elif mode == "CS2":
            result_J = self.CS2()
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))
        return plotc, result_J
    
    
    def clamp(self, plotc, dynamic_range, max_crop):

        plotc_max = np.max(plotc) - max_crop
        plotc_min = plotc_max - dynamic_range
        plotc_clamped = np.clip(plotc, plotc_min, plotc_max)
        return plotc_clamped, [plotc_min, plotc_max]

    def plot(self, plotc, dynamic_range=5, max_crop = 0, plot_mesh = True, opacity=None):
        """
        Plot the sound field in 3D.
        """
        plotc_clamped, p_range = self.clamp(plotc, dynamic_range, max_crop)
        
        if plot_mesh:
            if self.pv_mesh is None:
                self.pv_mesh = self.gen_mesh(self.pcd)
            cloud = pv.PolyData(self.pv_mesh)
        else:
            cloud = pv.PolyData(self.points)

        cloud["Sound Pressure(dB)"] = plotc_clamped  # Adding scalar values to the point cloud

        # Create a Plotter object and add the point cloud
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, cmap='rainbow', scalars='Sound Pressure(dB)',show_scalar_bar = True, point_size = 6, opacity=opacity)
        # Show the point cloud
        return plotter, p_range
    
    
    def gen_mesh(self, pcd):
        
        # variables for ball size, [lowest dimention, highest dimention, average dimention(starting point), increment, unit: m]

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        radii = [0.0005, 0.006, 0.001, 0.0005]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        
        # rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        # rec_mesh = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.001)

        vertices = np.asarray(rec_mesh.vertices)
        faces = np.asarray(rec_mesh.triangles)
        faces = np.c_[np.full(len(faces), 3), faces]  # Add a column of 3s indicating the number of points per face

        # Create PyVista PolyData from the vertices and faces
        pv_mesh = pv.PolyData(vertices, faces)
        return pv_mesh
    

    def save_temp_result(results_dict, freq, mode, result):
        key = f"{freq}_{mode}"
        if key in results_dict:
            # Find a unique key name by appending a number
            counter = 1
            new_key = f"{key}_{counter}"
            while new_key in results_dict:
                counter += 1
                new_key = f"{key}_{counter}"
            results_dict[new_key] = result
        else:
            results_dict[key] = result
        return results_dict

    def dbspl(self, plotc, magnitude=False, imag=False):
        """
        Convert sound pressure to dB SPL.
        """
        if imag:
            return 20 * np.log10(np.imag(plotc)/20e-6)
        if magnitude:
            return 20 * np.log10(np.real(plotc)/20e-6)
        else:
            return 20 * np.log10(np.abs(plotc / 20e-6))

    def temp_save():
        """save the result to a temporary file, the file name will be the parameter of the method
        file name: model + mode + freq
        filepath: temp/"""

        filepath = "temp/"
        
        


        return
    def result_save():
        return
    def DAS_muti_cal():
        return
    def CS_muti_cal():
        return
    def MUSIC_muti_val():
        return
