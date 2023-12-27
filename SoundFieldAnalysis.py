import numpy as np
import scipy.spatial
from scipy.signal import welch
import pyvista as pv
import cvxpy as cp
import open3d as o3d


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

    def __init__(self, mic_pcd, sound_field, pcd, freq=6000, seg_size=2000, nperseg=1600):
        self.mic_array = np.array(mic_pcd.points)
        self.sound_field = sound_field
        self.points = np.array(pcd.points)
        self.freq = freq
        self.seg_size = seg_size
        self.nperseg = nperseg
        self.S = 0
        self.Pxy = 0
        self.f = 0
        self.plotc = 0
        self.distance = 0
        self.pcd = pcd
        self.index = 0


    def DAS(self, points_subset):
        T = 10
        N = np.shape(self.sound_field)[1]
        fs = N / T
        m, _ = self.sound_field.shape
        f, Pxy = welch(self.sound_field, fs, noverlap=0, nperseg=self.nperseg, return_onesided=True, axis=-1)
        Pxy = fs / 1.28 * Pxy

        index = np.argmin(np.abs(f - self.freq))
        self.index = index
        C = 343
        omega = 2 * np.pi * f[index]
        k = omega / C

        Csm = np.zeros((m, m), dtype=complex)
        distance = scipy.spatial.distance.cdist(self.mic_array, points_subset, metric="euclidean")
        Vmn = np.zeros((m, m, len(points_subset)), dtype=complex)
        Vmn2 = np.zeros((m, m, len(points_subset)), dtype=complex)

        for i in range(m):
            for j in range(m):
                if i != j: # diangional removal
                    Csm[i, j] = np.sqrt(Pxy[i, index]) * np.sqrt(Pxy[j, index])
                    rm = distance[i]
                    rn = distance[j]
                    vm = np.exp(-1j * k * rm) / rm
                    vn = np.exp(-1j * k * rn) / rn
                    Vmn[i, j, :] = vm * np.conj(vn)
                    Vmn2[i, j, :] = np.abs(vm) ** 2 * np.abs(vn) ** 2

        self.S = Csm
        self.Pxy = Pxy
        self.f = f

        Jup = Csm[:, :, None] * Vmn
        result = 1 / np.sqrt(36*35) * (np.abs(Jup.sum(axis=(0, 1))) / np.sqrt(Vmn2.sum(axis=(0, 1))))
        return result, f

    def MUSIC(self):
        S = self.S
        eigVal, eigVec = np.linalg.eig(S)
        E_n = np.sqrt(eigVec[:,1:])*eigVal[1:]
        omega = 2*np.pi*self.f[self.index]
        C = 343
        k = omega/C
        distance = scipy.spatial.distance.cdist(self.mic_array, self.points, metric="euclidean")
        v = np.exp(-1j * k * distance) / distance
        P = []
        for i in range(len(distance[2,:])):
            a = v[:,i]
            Pi =  1/(np.abs(a.conj().T @ E_n @ E_n.conj().T @ a))**2
            P.append(Pi)
        return np.array(P)

    def CS(self):

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
        b = eigVec[:,0]*eigVal[0]
        # b = np.sqrt(self.Pxy[:,self.index])

        # eps = cp.Variable()
        eps = 1e-5
        x = cp.Variable(len(v[2,:]),  nonneg=True)
        # objective = cp.Minimize(cp.sum(x) + 10* eps)
        # constraints = [cp.sum_squares(v @ x - b) <= eps, eps >= 0]

        objective = cp.Minimize(cp.sum(x))
        # constraints = [cp.sum_squares(v @ x - b)<=eps/10]
        # constraints = [cp.norm(v.T@(b - v @ x))<=2]
        constraints = [1/2*cp.norm(b - v @ x)<=2]
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

    def calculate(self):
        split_points = np.array_split(self.points, np.round(self.points.shape[0] / self.seg_size))
        J = []
        for splitP in split_points:
            Js, _ = self.DAS(splitP)
            J.append(Js)
        result_J = np.concatenate(J, axis=0)
        return result_J

    def plot(self, mode = "DAS", dynamic_range=5, max_crop = 0):
        if mode == "DAS":
            result_J = self.calculate()
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))
        elif mode == "MUSIC":
            result_J = self.MUSIC()
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))
        elif mode == "CS":
            result_J = self.CS()
            plotc = 20 * np.log10(np.abs(result_J / 20e-6))

        self.plotc = plotc
        plotc_max = np.max(plotc) - max_crop
        plotc_min = plotc_max - dynamic_range
        plotc_clamped = np.clip(plotc, plotc_min, plotc_max)

        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        radii = [0.001, 0.005, 0.02, 0.04]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            self.pcd, o3d.utility.DoubleVector(radii))
        
        vertices = np.asarray(rec_mesh.vertices)
        faces = np.asarray(rec_mesh.triangles)
        faces = np.c_[np.full(len(faces), 3), faces]  # Add a column of 3s indicating the number of points per face

        # Create PyVista PolyData from the vertices and faces
        pv_mesh = pv.PolyData(vertices, faces)

        
        # Assuming self.points are your point coordinates and plotc is the scalar value for each point
        # cloud = pv.PolyData(self.points)
        cloud = pv.PolyData(pv_mesh)
        cloud["Sound Pressure(dB)"] = plotc_clamped  # Adding scalar values to the point cloud

        # Create a Plotter object and add the point cloud
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, cmap='rainbow', scalars='Sound Pressure(dB)',show_scalar_bar = True, point_size = 6, lighting = True, specular = True, specular_power = 100)
        plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.1)

        # Show the point cloud
        plotter.show()

    def plot_and_save():
        return
    def CS_muti_cal():
        return
    def MUSIC_muti_val():
        return
