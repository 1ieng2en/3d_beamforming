import numpy as np
import scipy.spatial
from scipy.signal import welch
import pyvista as pv

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
        self.S = None

    def DAS(self, points_subset):
        T = 10
        N = np.shape(self.sound_field)[1]
        fs = N / T
        m, _ = self.sound_field.shape
        f, Pxy = welch(self.sound_field, fs, noverlap=0, nperseg=self.nperseg, return_onesided=True, axis=-1)
        Pxy = fs / 1.28 * Pxy

        index = np.argmin(np.abs(f - self.freq))
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

        Jup = Csm[:, :, None] * Vmn
        self.S = Jup
        result = 1 / np.sqrt(36*35) * (np.abs(Jup.sum(axis=(0, 1))) / np.sqrt(Vmn2.sum(axis=(0, 1))))
        return result, f

    def MUSIC(self):
        S = self.S
        eigVal, eigVec = np.linalg.eig(S)
        

        return eigVal, eigVec


    def calculate_3D_DAS(self):
        split_points = np.array_split(self.points, np.round(self.points.shape[0] / self.seg_size))
        J = []
        for splitP in split_points:
            Js, _ = self.DAS(splitP)
            J.append(Js)
        result_J = np.concatenate(J, axis=0)
        return result_J

    def plot(self, dynamic_range=5):
        result_J = self.calculate_3D_DAS()
        plotc = 20 * np.log10(np.abs(result_J / 20e-6))
        plotc_max = np.max(plotc)
        plotc_min = plotc_max - dynamic_range
        plotc_clamped = np.clip(plotc, plotc_min, plotc_max)

        # Assuming self.points are your point coordinates and plotc is the scalar value for each point
        cloud = pv.PolyData(self.points)
        cloud["Sound Pressure(dB)"] = plotc_clamped  # Adding scalar values to the point cloud

        # Create a Plotter object and add the point cloud
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, cmap='rainbow', scalars='Sound Pressure(dB)', render_points_as_spheres=True, point_size=8)
        

        # Show the point cloud
        plotter.show()
