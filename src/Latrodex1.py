

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d


class Grid:
    """Class for handling interpolation between two curves."""
    
    def __init__(self, curve1, curve2, smoothing=0.0005, num_points=1000):
        """
        Initialize the grid_instance with two curves.
        
        Parameters:
        -----------
        curve1 : np.ndarray
            First curve points as (x, y) coordinates
        curve2 : np.ndarray
            Second curve points as (x, y) coordinates
        smoothing : float, optional
            Smoothing factor for spline interpolation. Default is 0.0005.
        num_points : int, optional
            Number of points to use for high-resolution sampling. Default is 1000.
        """
        self.x1_raw, self.y1_raw = curve1[:, 0], curve1[:, 1]
        self.x2_raw, self.y2_raw = curve2[:, 0], curve2[:, 1]
        self.smoothing = smoothing
        self.num_points = num_points
        
        # Apply spline smoothing
        tck1, u1 = splprep([self.x1_raw, self.y1_raw], s=self.smoothing)
        tck2, u2 = splprep([self.x2_raw, self.y2_raw], s=self.smoothing)
        
        # Evaluate the smoothed spline at more refined parameter values
        u1_new = np.linspace(u1.min(), u1.max(), self.num_points)
        u2_new = np.linspace(u2.min(), u2.max(), self.num_points)
        self.x1, self.y1 = splev(u1_new, tck1)
        self.x2, self.y2 = splev(u2_new, tck2)
        
        # Create high-resolution parameter values
        self.xi_values_high_res = np.linspace(0, 1, self.num_points)
        self.zi_values_high_res = np.linspace(0, 1, self.num_points)
        
        # Precompute curve properties
        self._precompute_curve_properties()
    
    @staticmethod
    def discretize(x, y, xi_values):
        """
        Discretize a curve according to arc length.
        
        Parameters:
        -----------
        x, y : np.ndarray
            Curve coordinates
        xi_values : np.ndarray
            Parameter values to evaluate at
            
        Returns:
        --------
        x_resampled, y_resampled : np.ndarray
            Resampled curve coordinates
        """
        dr = (np.diff(x)**2 + np.diff(y)**2)**0.5  # segment lengths
        r = np.zeros_like(x)
        r[1:] = np.cumsum(dr)  # integrate path
        total_length = r[-1]
        xi_values_scaled = xi_values * total_length  # Scale xi values to total arc length
        x_resampled = np.interp(xi_values_scaled, r, x)  # interpolate x at scaled xi values
        y_resampled = np.interp(xi_values_scaled, r, y)  # interpolate y at scaled xi values
        return x_resampled, y_resampled
    
    @staticmethod
    def compute_curve_properties(x, y, xi_values):
        """
        Compute geometric properties of a curve.
        
        Parameters:
        -----------
        x, y : np.ndarray
            Curve coordinates
        xi_values : np.ndarray
            Parameter values to evaluate at
            
        Returns:
        --------
        Tuple containing:
            x_resampled, y_resampled : np.ndarray
                Resampled curve coordinates
            unit_tangent_x, unit_tangent_y : np.ndarray
                Tangent vector components
            curvature : np.ndarray
                Curvature values
            unit_normal_x, unit_normal_y : np.ndarray
                Normal vector components
        """
        x_resampled, y_resampled = Grid.discretize(x, y, xi_values)
        
        dx_dxi = np.gradient(x_resampled, xi_values)
        dy_dxi = np.gradient(y_resampled, xi_values)
        
        d2x_dxi2 = np.gradient(dx_dxi, xi_values)
        d2y_dxi2 = np.gradient(dy_dxi, xi_values)
        
        g11 = dx_dxi**2 + dy_dxi**2
        g11 = np.maximum(g11, 1e-12)
        curvature = np.abs(d2x_dxi2 * dy_dxi - d2y_dxi2 * dx_dxi) / np.power(g11, 1.5)
        
        unit_tangent_x = dx_dxi / np.sqrt(g11)
        unit_tangent_y = dy_dxi / np.sqrt(g11)

        unit_normal_x = -dy_dxi / np.sqrt(g11)
        unit_normal_y = dx_dxi / np.sqrt(g11)
        
        return x_resampled, y_resampled, unit_tangent_x, unit_tangent_y, curvature, unit_normal_x, unit_normal_y
    
    def _precompute_curve_properties(self):
        """Precompute properties for both curves."""
        result1 = self.compute_curve_properties(self.x1, self.y1, self.xi_values_high_res)
        result2 = self.compute_curve_properties(self.x2, self.y2, self.zi_values_high_res)
        
        self.x1_resampled, self.y1_resampled = result1[0], result1[1]
        self.unit_tangent_1_x, self.unit_tangent_1_y = result1[2], result1[3]
        self.curvature_1 = result1[4]
        self.unit_normal_1_x, self.unit_normal_1_y = result1[5], result1[6]
        
        self.x2_resampled, self.y2_resampled = result2[0], result2[1]
        self.unit_tangent_2_x, self.unit_tangent_2_y = result2[2], result2[3]
        self.curvature_2 = result2[4]
        self.unit_normal_2_x, self.unit_normal_2_y = result2[5], result2[6]
    
    def find_corresponding_z(self, xi):
        """
        Find the corresponding z parameter on the second curve for a given xi on the first curve.
        
        Parameters:
        -----------
        xi : float
            Parameter value on the first curve
            
        Returns:
        --------
        Tuple containing:
            x1_xi, y1_xi : float
                Point on the first curve at xi
            x2_z, y2_z : float
                Corresponding point on the second curve
            z : float
                Parameter value on the second curve
        """
        def objective(z):
            idx_xi = np.searchsorted(self.xi_values_high_res, xi)
            x1_xi = self.x1_resampled[idx_xi]
            y1_xi = self.y1_resampled[idx_xi]
            t1x_xi = self.unit_tangent_1_x[idx_xi]
            t1y_xi = self.unit_tangent_1_y[idx_xi]
            
            idx_z = np.searchsorted(self.zi_values_high_res, z)
            x2_z = self.x2_resampled[idx_z]
            y2_z = self.y2_resampled[idx_z]
            t2x_z = self.unit_tangent_2_x[idx_z]
            t2y_z = self.unit_tangent_2_y[idx_z]
            
            c_x = x2_z - x1_xi
            c_y = y2_z - y1_xi
            
            dot_product = c_x * (t1x_xi + t2x_z) + c_y * (t1y_xi + t2y_z)
            return abs(dot_product) + (c_x**2 + c_y**2)  # Distance term added
        
        try:
            result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
            z = result.x
            idx_xi = np.searchsorted(self.xi_values_high_res, xi)
            idx_z = np.searchsorted(self.zi_values_high_res, z)
            x1_xi = self.x1_resampled[idx_xi]
            y1_xi = self.y1_resampled[idx_xi]
            x2_z = self.x2_resampled[idx_z]
            y2_z = self.y2_resampled[idx_z]
            return x1_xi, y1_xi, x2_z, y2_z, z
        except ValueError:
            # If the minimization algorithm fails to converge, return None
            return None, None, None, None, None
            
    def find_corresponding_xi(self, z):
        """
        Find the corresponding xi parameter on the first curve for a given z on the second curve.
        This is the reverse direction matching function.
        
        Parameters:
        -----------
        z : float
            Parameter value on the second curve
            
        Returns:
        --------
        Tuple containing:
            x1_xi, y1_xi : float
                Point on the first curve
            x2_z, y2_z : float
                Corresponding point on the second curve at z
            xi : float
                Parameter value on the first curve
        """
        def objective(xi):
            idx_xi = np.searchsorted(self.xi_values_high_res, xi)
            x1_xi = self.x1_resampled[idx_xi]
            y1_xi = self.y1_resampled[idx_xi]
            t1x_xi = self.unit_tangent_1_x[idx_xi]
            t1y_xi = self.unit_tangent_1_y[idx_xi]
            
            idx_z = np.searchsorted(self.zi_values_high_res, z)
            x2_z = self.x2_resampled[idx_z]
            y2_z = self.y2_resampled[idx_z]
            t2x_z = self.unit_tangent_2_x[idx_z]
            t2y_z = self.unit_tangent_2_y[idx_z]
            
            c_x = x1_xi - x2_z  # Note the reversed order compared to find_corresponding_z
            c_y = y1_xi - y2_z
            
            dot_product = c_x * (t2x_z + t1x_xi) + c_y * (t2y_z + t1y_xi)
            return abs(dot_product) + (c_x**2 + c_y**2)
        
        try:
            result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
            xi = result.x
            idx_xi = np.searchsorted(self.xi_values_high_res, xi)
            idx_z = np.searchsorted(self.zi_values_high_res, z)
            x1_xi = self.x1_resampled[idx_xi]
            y1_xi = self.y1_resampled[idx_xi]
            x2_z = self.x2_resampled[idx_z]
            y2_z = self.y2_resampled[idx_z]
            return x1_xi, y1_xi, x2_z, y2_z, xi
        except ValueError:
            # If the minimization algorithm fails to converge, return None
            return None, None, None, None, None
    
    # === Filtering, Gap Filling, and Smoothing Methods ===
    def filter_results(self, pairs, min_diff):
        """
        Filter correspondence pairs to remove redundant or invalid matches.
        
        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        min_diff : float
            Minimum difference between consecutive points
            
        Returns:
        --------
        np.ndarray
            Filtered array of [xi, z] pairs
        """
        filtered = []
        for i in range(len(pairs)):
            if i == 0:
                filtered.append(pairs[i])
            else:
                prev = pairs[i - 1]
                curr = pairs[i]
                cond_1 = curr[0] > prev[0]
                cond_2 = curr[1] > prev[1]
                diff_cond = abs(curr[0] - prev[0]) > min_diff or abs(curr[1] - prev[1]) > min_diff
                if cond_1 and cond_2 and diff_cond:
                    filtered.append(curr)
        return np.array(filtered)
    
    def fill_gaps_with_matching(self, pairs, resolution, threshold):
        """
        Fill gaps in the correspondence pairs by computing additional matches.
        
        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        resolution : float
            Minimum resolution between points
        threshold : float
            Maximum chord length for valid matches
            
        Returns:
        --------
        np.ndarray
            Array with gaps filled
        """
        pairs = sorted(pairs, key=lambda x: x[0])
        new_points = np.empty((0, 2))
        for i in range(len(pairs) - 1):
            if (abs(pairs[i+1][0] - pairs[i][0]) > resolution or
                abs(pairs[i+1][1] - pairs[i][1]) > resolution):
                xi_mid = (pairs[i][0] + pairs[i+1][0]) / 2
                x1_xi, y1_xi, x2_z, y2_z, z = self.find_corresponding_z(xi_mid)
                if z is not None:
                    chord = np.linalg.norm([x2_z - x1_xi, y2_z - y1_xi])
                    if chord < threshold:
                        new_points = np.vstack([new_points, [xi_mid, z]])
        return np.array(sorted(np.vstack([pairs, new_points]), key=lambda x: x[0]))
    
    def find_gaps_xi(self, pairs, resolution):
        """
        Find gaps in the xi direction.
        
        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        resolution : float
            Minimum resolution between points
            
        Returns:
        --------
        list
            List of xi values where gaps exist
        """
        pairs = sorted(pairs, key=lambda x: x[0])
        gaps = []
        for i in range(len(pairs) - 1):
            if (abs(pairs[i+1][0] - pairs[i][0]) > resolution or
                abs(pairs[i+1][1] - pairs[i][1]) > resolution):
                gaps.append((pairs[i+1][0] + pairs[i][0]) / 2)
        return gaps
    
    def find_gaps_z(self, pairs, resolution):
        """
        Find gaps in the z direction.
        
        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        resolution : float
            Minimum resolution between points
            
        Returns:
        --------
        list
            List of z values where gaps exist
        """
        pairs = sorted(pairs, key=lambda x: x[1])
        gaps = []
        for i in range(len(pairs) - 1):
            if (abs(pairs[i+1][0] - pairs[i][0]) > resolution or
                abs(pairs[i+1][1] - pairs[i][1]) > resolution):
                gaps.append((pairs[i+1][1] + pairs[i][1]) / 2)
        return gaps
    
    def update_xi(self, pairs, xi_gaps, threshold):
        """
        Update pairs by adding new xi correspondences.
        
        Parameters:
        -----------
        pairs : list or np.ndarray
            List of [xi, z] pairs
        xi_gaps : list
            List of xi values where gaps exist
        threshold : float
            Maximum chord length for valid matches
            
        Returns:
        --------
        np.ndarray
            Updated array of [xi, z] pairs
        """
        updated = pairs.copy()
        for xi in xi_gaps:
            x1_xi, y1_xi, x2_z, y2_z, z = self.find_corresponding_z(xi)
            if z is not None:
                chord = np.linalg.norm([x2_z - x1_xi, y2_z - y1_xi])
                if chord < threshold:
                    updated.append([xi, z])
        return np.array(sorted(updated, key=lambda x: x[0]))
    
    def update_z(self, pairs, z_gaps, threshold):
        """
        Update pairs by adding new z correspondences using reverse direction matching.
        
        Parameters:
        -----------
        pairs : list or np.ndarray
            List of [xi, z] pairs
        z_gaps : list
            List of z values where gaps exist
        threshold : float
            Maximum chord length for valid matches
            
        Returns:
        --------
        np.ndarray
            Updated array of [xi, z] pairs
        """
        updated = pairs.copy()
        for z in z_gaps:
            x1_xi, y1_xi, x2_z, y2_z, xi = self.find_corresponding_xi(z)
            if xi is not None:
                chord = np.linalg.norm([x2_z - x1_xi, y2_z - y1_xi])
                if chord < threshold:
                    updated.append([xi, z])
        return np.array(sorted(updated, key=lambda x: x[0]))
    
    def laplacian_1d(self, pairs, alpha=0.1, iterations=50):
        """
        Apply Laplacian smoothing to the correspondence pairs.
        
        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        alpha : float, optional
            Smoothing factor. Default is 0.1.
        iterations : int, optional
            Number of smoothing iterations. Default is 50.
            
        Returns:
        --------
        np.ndarray
            Smoothed array of [xi, z] pairs
        """
        x = pairs[:, 0].copy()
        z = pairs[:, 1].copy()

        for _ in range(iterations):
            x[1:-1] += alpha * (x[0:-2] + x[2:] - 2 * x[1:-1])
            z[1:-1] += alpha * (z[0:-2] + z[2:] - 2 * z[1:-1])

        return np.column_stack([x, z])
    
    def compute_correspondences(self, num_points=30, start=0.01, end=0.99, 
                                threshold=1e-3, resolution=0.01, filter_min=5e-4,
                                smooth_alpha=0.5, smooth_iterations=10, smooth=True):
        """
        Compute corresponding points between the two curves with optional filtering and smoothing.

        Parameters:
        -----------
        num_points : int
            Number of correspondence points.
        start : float
            Starting parameter value.
        end : float
            Ending parameter value.
        threshold : float
            Maximum chord length for valid matches.
        resolution : float
            Minimum resolution between points.
        filter_min : float
            Minimum difference for filtering.
        smooth_alpha : float
            Smoothing factor.
        smooth_iterations : int
            Number of smoothing iterations.
        smooth : bool
            Whether to apply smoothing at the end.

        Returns:
        --------
        xi_list, z_list : np.ndarray
            Parameter values on the first and second curve.
        """
        xi_values = np.linspace(start, end, num_points)

        xi_list = []
        z_list = []

        for xi in xi_values:
            x1_xi, y1_xi, x2_z, y2_z, z = self.find_corresponding_z(xi)
            if z is not None:
                xi_list.append(xi)
                z_list.append(z)

        pairs = np.column_stack([xi_list, z_list])
        pairs = self.filter_results(pairs, filter_min)
        pairs = self.fill_gaps_with_matching(pairs, resolution, threshold)
        pairs = self.filter_results(pairs, filter_min)

        xi_gaps = self.find_gaps_xi(pairs, resolution)
        pairs = self.update_xi(pairs.tolist(), xi_gaps, threshold)

        z_gaps = self.find_gaps_z(pairs, resolution)
        pairs = self.update_z(pairs.tolist(), z_gaps, threshold)

        pairs = self.filter_results(pairs, filter_min)
        if smooth:
            pairs = self.laplacian_1d(pairs, alpha=smooth_alpha, iterations=smooth_iterations)

        self.correspondence_points = {
            'pairs': pairs.copy(),
            'xi': pairs[:, 0].copy(),
            'z':  pairs[:, 1].copy()
        }

        return self.correspondence_points['xi'], self.correspondence_points['z']



    def generate_grid_from_correspondence(self, num_s_values=20):
        """
        Use current self.correspondence_points to generate a grid.

        Parameters:
        -----------
        num_s_values : int
            Number of s-levels (cross-sections)

        Returns:
        --------
        x, y : np.ndarray
            Grid arrays (xi × s)
        """
        if not hasattr(self, 'correspondence_points'):
            raise RuntimeError("No correspondence_points found.")

        xi = self.correspondence_points['xi']
        z = self.correspondence_points['z']
        rvecs_list = [self.interpolate(xi, z, s) for s in np.linspace(0, 1, num_s_values)]

        ni, nj = len(xi), num_s_values
        x = np.zeros((ni, nj))
        y = np.zeros((ni, nj))
        for j, rvecs in enumerate(rvecs_list):
            x[:, j] = rvecs[:, 0]
            y[:, j] = rvecs[:, 1]
        return x, y

    def smooth_correspondence_param(self, param='z', alpha=0.1, iterations=10, monotone=False):
        """
        Smooth xi or z in self.correspondence_points after matching.

        Parameters:
        -----------
        param : str
            'xi' or 'z'
        alpha : float
            Smoothing strength
        iterations : int
            Number of iterations
        monotone : bool
            Enforce monotonic increase
        """
        arr = self.correspondence_points[param].copy()
        for _ in range(iterations):
            arr[1:-1] += alpha * (arr[0:-2] + arr[2:] - 2 * arr[1:-1])
            if monotone:
                delta = np.diff(arr)
                delta[delta < 0] = 0
                arr = np.concatenate([[arr[0]], arr[0] + np.cumsum(delta)])
        self.correspondence_points[param] = arr
        self.correspondence_points['pairs'] = np.column_stack([
            self.correspondence_points['xi'],
            self.correspondence_points['z']
        ])


    def generate_interpolated_curves(self, num_s_values=10):
        """
        Generate a set of interpolated curves between the two input curves.
        
        Parameters:
        -----------
        num_s_values : int, optional
            Number of intermediate curves to generate. Default is 10.
            
        Returns:
        --------
        r_vectors_list : list of np.ndarray
            List of interpolated curves
        s_values : list or np.ndarray
            Interpolation parameter values
        """
        if not hasattr(self, 'correspondence_points'):
            self.compute_correspondences()
            
        s_values = np.linspace(0, 1, num_s_values)
        r_vectors_list = [
            self.interpolate(
                self.correspondence_points['xi'], 
                self.correspondence_points['z'], 
                s
            ) for s in s_values
        ]
        
        return r_vectors_list, s_values
    def ortho_grid(self, x, y, iterations=2, omega=0.1):
        """
        Orthogonalize a grid using an elliptic PDE-based smoothing approach.
        """
        for _ in range(iterations):
            dx_dxi  = (x[2:, 1:-1] - x[:-2, 1:-1]) * 0.5
            dx_deta = (x[1:-1, 2:] - x[1:-1, :-2]) * 0.5
            dy_dxi  = (y[2:, 1:-1] - y[:-2, 1:-1]) * 0.5
            dy_deta = (y[1:-1, 2:] - y[1:-1, :-2]) * 0.5
            
            alpha = dx_deta**2 + dy_deta**2
            beta  = dx_dxi * dx_deta + dy_dxi * dy_deta
            gamma = dx_dxi**2 + dy_dxi**2
            
            d2x_dxi2   = x[2:, 1:-1] - 2 * x[1:-1, 1:-1] + x[:-2, 1:-1]
            d2x_deta2  = x[1:-1, 2:] - 2 * x[1:-1, 1:-1] + x[1:-1, :-2]
            d2y_dxi2   = y[2:, 1:-1] - 2 * y[1:-1, 1:-1] + y[:-2, 1:-1]
            d2y_deta2  = y[1:-1, 2:] - 2 * y[1:-1, 1:-1] + y[1:-1, :-2]
            d2x_dxideta = (x[2:, 2:] - x[:-2, 2:] - x[2:, :-2] + x[:-2, :-2]) * 0.25
            d2y_dxideta = (y[2:, 2:] - y[:-2, 2:] - y[2:, :-2] + y[:-2, :-2]) * 0.25
            
            x[1:-1, 1:-1] += omega * (alpha * d2x_dxi2 + 2 * beta * d2x_dxideta + gamma * d2x_deta2)
            y[1:-1, 1:-1] += omega * (alpha * d2y_dxi2 + 2 * beta * d2y_dxideta + gamma * d2y_deta2)
        
        return x, y
    def smooth_single_param(self, param='z', alpha=0.1, iterations=50, monotone=True):
        """
        Apply smoothing to either 'xi' or 'z' during runtime.

        Parameters:
        -----------
        param : str
            Which parameter to smooth: 'xi' or 'z'
        alpha : float
            Smoothing factor
        iterations : int
            Number of iterations
        monotone : bool
            Whether to enforce weak monotonicity (no folds)

        Returns:
        --------
        np.ndarray
            Smoothed values
        """
        if not hasattr(self, 'correspondence_points'):
            raise RuntimeError("Correspondences not computed yet.")

        data = self.correspondence_points[param].copy()

        for _ in range(iterations):
            # 1D Laplacian smoothing
            data[1:-1] += alpha * (data[0:-2] + data[2:] - 2 * data[1:-1])
            if monotone:
                # Enforce non-decreasing trend without forcing global max
                delta = np.diff(data)
                delta[delta < 0] = 0
                data = np.concatenate([[data[0]], data[0] + np.cumsum(delta)])

        self.correspondence_points[param] = data
        self.correspondence_points['pairs'] = np.column_stack([
            self.correspondence_points['xi'], self.correspondence_points['z']
        ])
        return data
    #Another Version
    # def smooth_single_param(self, param='z', alpha=0.1, iterations=50, monotone=True):
        # """
        # Apply smoothing to either 'xi' or 'z' during runtime.
    
        # Parameters:
        # -----------
        # param : str
            # Which parameter to smooth: 'xi' or 'z'
        # alpha : float
            # Smoothing factor
        # iterations : int
            # Number of iterations
        # monotone : bool
            # Whether to enforce monotonicity
    
        # Returns:
        # --------
        # np.ndarray
            # Smoothed values
        # """
        # if not hasattr(self, 'correspondence_points'):
            # raise RuntimeError("Correspondences not computed yet.")
    
        # data = self.correspondence_points[param].copy()
        # for _ in range(iterations):
            # data[1:-1] += alpha * (data[0:-2] + data[2:] - 2 * data[1:-1])
            # if monotone:
                # data = np.maximum.accumulate(data)
    
        # self.correspondence_points[param] = data
        # self.correspondence_points['pairs'] = np.column_stack([
            # self.correspondence_points['xi'], self.correspondence_points['z']
        # ])
        # return data

    def smooth_grid(self, x, y, alpha=0.1, iterations=10):
        """
        Apply 2D Laplacian smoothing to the full grid.

        Parameters:
        -----------
        x, y : np.ndarray
            Grid coordinates to be smoothed
        alpha : float
            Smoothing weight
        iterations : int
            Number of smoothing iterations

        Returns:
        --------
        Smoothed x, y arrays
        """
        x = x.copy()
        y = y.copy()
        for _ in range(iterations):
            x_inner = x[1:-1, 1:-1]
            y_inner = y[1:-1, 1:-1]
            lap_x = (x[0:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, 0:-2] + x[1:-1, 2:] - 4 * x_inner)
            lap_y = (y[0:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, 0:-2] + y[1:-1, 2:] - 4 * y_inner)
            x[1:-1, 1:-1] += alpha * lap_x
            y[1:-1, 1:-1] += alpha * lap_y
        return x, y

    def interpolate(self, xi_values, zi_values, s):
        """
        Interpolate between the two curves at parameter s.
        
        Parameters:
        -----------
        xi_values : list or np.ndarray
            Parameter values on the first curve
        zi_values : list or np.ndarray
            Parameter values on the second curve
        s : float
            Interpolation parameter between 0 and 1
            
        Returns:
        --------
        r_vectors : np.ndarray
            Interpolated points
        """
        r_vectors = []
        for xi, zi in zip(xi_values, zi_values):
            # Find the indices of xi and zi in the precomputed lists
            xi_idx = np.searchsorted(self.xi_values_high_res, xi)
            zi_idx = np.searchsorted(self.zi_values_high_res, zi)

            # Retrieve the precomputed values for the given xi and zi
            x1_interp_val = self.x1_resampled[xi_idx]
            y1_interp_val = self.y1_resampled[xi_idx]
            x2_interp_val = self.x2_resampled[zi_idx]
            y2_interp_val = self.y2_resampled[zi_idx]

            unit_tangent_1_x_val = self.unit_tangent_1_x[xi_idx]
            unit_tangent_1_y_val = self.unit_tangent_1_y[xi_idx]
            unit_normal_1_x_val = self.unit_normal_1_x[xi_idx]
            unit_normal_1_y_val = self.unit_normal_1_y[xi_idx]

            unit_tangent_2_x_val = self.unit_tangent_2_x[zi_idx]
            unit_tangent_2_y_val = self.unit_tangent_2_y[zi_idx]

            # Calculate theta using arctan2
            theta = np.arctan2(unit_tangent_2_y_val, unit_tangent_2_x_val) - np.arctan2(unit_tangent_1_y_val, unit_tangent_1_x_val)
            
            # Ensure theta is in the range [-pi, pi]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            chord_length = np.linalg.norm([x2_interp_val - x1_interp_val, y2_interp_val - y1_interp_val])

            # if np.isclose(theta, 0.0, atol=1e-10):
                # direction = np.array([unit_normal_1_x_val, unit_normal_1_y_val])
                # r_vector = np.array([x1_interp_val, y1_interp_val]) + s * chord_length * direction
                
            if np.isclose(theta, 0.0, atol=1e-10):
                # Correct linear interpolation between the endpoints
                r_vector = (1 - s) * np.array([x1_interp_val, y1_interp_val]) \
                           + s * np.array([x2_interp_val, y2_interp_val])

            else:
                sin_half_theta = np.sin(0.5 * theta)
                if np.isclose(sin_half_theta, 0.0, atol=1e-4):
                    sin_half_theta = 1
                
                r_vector = np.array([x1_interp_val, y1_interp_val]) + \
                       chord_length * np.sin(0.5 * s * theta) / sin_half_theta * \
                       (np.cos(0.5 * s * theta) * np.array([unit_normal_1_x_val, unit_normal_1_y_val]) -
                        np.sin(0.5 * s * theta) * np.array([unit_tangent_1_x_val, unit_tangent_1_y_val]))
            
            r_vectors.append(r_vector)

        return np.array(r_vectors)
    
    def plot_results(self, r_vectors_list, s_values, original_curves=None, figsize=(20, 10)):
        plt.figure(figsize=figsize)
        
        for i, r_vectors in enumerate(r_vectors_list):
            plt.plot(r_vectors[:, 0], r_vectors[:, 1], 'blue', marker='', 
                     linestyle='-', label=f's={s_values[i]:.2f}', linewidth=1)

        num_points = len(r_vectors_list[0])
        for i in range(num_points):
            x_points = [r_vectors[i, 0] for r_vectors in r_vectors_list]
            y_points = [r_vectors[i, 1] for r_vectors in r_vectors_list]
            
            plt.plot(x_points, y_points, 'blue', linestyle='-', linewidth=1)
            
        plt.plot(self.x1, self.y1, 'black', linewidth=2)
        plt.plot(self.x2, self.y2, 'black', linewidth=2)
        
        if original_curves is not None:
            for curve in original_curves:
                plt.plot(curve[:, 0], curve[:, 1], 'red', linewidth=2)
        
        plt.axis('equal')
        plt.axis('off')
        
        return plt
    def plot_grid(self, gx, gy, ax=None, title=None, color='b-', linewidth=0.5, label_parameters=False):
        """
        Plot a grid with optional arc-length parameter labels on boundary curves.

        Parameters:
        -----------
        gx, gy : np.ndarray
            Grid arrays (shape: [ni, nj])
        ax : matplotlib.axes._axes.Axes or None
            Axes to plot on; if None, creates new figure
        title : str or None
            Title of the subplot
        color : str
            Grid line color/style
        linewidth : float
            Width of the grid lines
        label_parameters : bool
            If True, label xi and z values on the curves
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        if title:
            ax.set_title(title)

        ax.plot(self.x1, self.y1, 'k-', linewidth=2)
        ax.plot(self.x2, self.y2, 'k-', linewidth=2)

        for i in range(gx.shape[0]):
            ax.plot(gx[i, :], gy[i, :], color, linewidth=linewidth)
        for j in range(gx.shape[1]):
            ax.plot(gx[:, j], gy[:, j], color, linewidth=linewidth)

        if label_parameters and hasattr(self, 'xi_values_high_res') and hasattr(self, 'zi_values_high_res'):
            skip = max(len(self.x1_resampled) // 30, 1)
            for i in range(0, len(self.x1_resampled), skip):
                ax.text(self.x1_resampled[i], self.y1_resampled[i], f"{self.xi_values_high_res[i]:.2f}",
                        fontsize=9, ha='right', va='bottom', color='red')
            for i in range(0, len(self.x2_resampled), skip):
                ax.text(self.x2_resampled[i], self.y2_resampled[i], f"{self.zi_values_high_res[i]:.2f}",
                        fontsize=9, ha='left', va='top', color='red')

        ax.axis('equal')
        ax.axis('off')

    def lagrangian_smooth_grid(self, x, y, weight_self=0.5, iterations=10):
        x_s, y_s = x.copy(), y.copy()
        w_n = (1.0 - weight_self) / 4.0
        for _ in range(iterations):
            x_new = x_s.copy()
            y_new = y_s.copy()
            x_new[1:-1, 1:-1] = (
                weight_self * x_s[1:-1, 1:-1]
                + w_n * (x_s[2:, 1:-1] + x_s[:-2, 1:-1] + x_s[1:-1, 2:] + x_s[1:-1, :-2])
            )
            y_new[1:-1, 1:-1] = (
                weight_self * y_s[1:-1, 1:-1]
                + w_n * (y_s[2:, 1:-1] + y_s[:-2, 1:-1] + y_s[1:-1, 2:] + y_s[1:-1, :-2])
            )
            x_s, y_s = x_new, y_new
        return x_s, y_s

    def extrapolate_boundary(self, x, y, direction='i', side='min', layers=1):
        x_ext, y_ext = x.copy(), y.copy()

        if direction == 'i':
            if side == 'min':
                dx = x[1, :] - x[0, :]
                dy = y[1, :] - y[0, :]
                new_x = [x[0, :] - (i + 1) * dx for i in reversed(range(layers))]
                new_y = [y[0, :] - (i + 1) * dy for i in reversed(range(layers))]
                x_ext = np.vstack(new_x + [x])
                y_ext = np.vstack(new_y + [y])
            elif side == 'max':
                dx = x[-1, :] - x[-2, :]
                dy = y[-1, :] - y[-2, :]
                new_x = [x[-1, :] + (i + 1) * dx for i in range(layers)]
                new_y = [y[-1, :] + (i + 1) * dy for i in range(layers)]
                x_ext = np.vstack([x] + new_x)
                y_ext = np.vstack([y] + new_y)

        elif direction == 'j':
            if side == 'min':
                dx = x[:, 1] - x[:, 0]
                dy = y[:, 1] - y[:, 0]
                new_x = [x[:, 0] - (i + 1) * dx for i in reversed(range(layers))]
                new_y = [y[:, 0] - (i + 1) * dy for i in reversed(range(layers))]
                x_ext = np.hstack([np.column_stack(new_x), x])
                y_ext = np.hstack([np.column_stack(new_y), y])
            elif side == 'max':
                dx = x[:, -1] - x[:, -2]
                dy = y[:, -1] - y[:, -2]
                new_x = [x[:, -1] + (i + 1) * dx for i in range(layers)]
                new_y = [y[:, -1] + (i + 1) * dy for i in range(layers)]
                x_ext = np.hstack([x, np.column_stack(new_x)])
                y_ext = np.hstack([y, np.column_stack(new_y)])

        return x_ext, y_ext


    def nonuniform_resolution(intervals):
        """
        Generate a nonuniform parameter array from a list of (start, stop, step) tuples.

        Parameters:
        -----------
        intervals : list of tuples
            Each tuple is (start, stop, step)

        Returns:
        --------
        np.ndarray
            Concatenated parameter array
        """
        arrays = []
        for i, (start, stop, step) in enumerate(intervals):
            arr = np.arange(start, stop, step)
            if i > 0 and np.isclose(arr[0], arrays[-1][-1]):
                arr = arr[1:]  # skip duplicate start
            arrays.append(arr)
        return np.unique(np.concatenate(arrays))
    def nonuniform_sweep(self, s_intervals):
        """
        Generate a grid using nonuniform spacing in the s-direction defined by intervals.

        Parameters:
        -----------
        s_intervals : list of tuples
            Each tuple is (start, stop, step) for s values

        Returns:
        --------
        x, y : np.ndarray
            Grid coordinates (xi × s)
        """
        # Generate s_values with deduplication at boundaries
        s_values = []
        for i, (start, stop, step) in enumerate(s_intervals):
            arr = np.arange(start, stop, step)
            if i > 0 and np.isclose(arr[0], s_values[-1][-1]):
                arr = arr[1:]
            s_values.append(arr)
        s_values = np.concatenate(s_values)

        xi = self.correspondence_points['xi']
        z = self.correspondence_points['z']

        rvecs_list = [self.interpolate(xi, z, s) for s in s_values]
        ni, nj = len(xi), len(s_values)

        x = np.zeros((ni, nj))
        y = np.zeros((ni, nj))
        for j, rvecs in enumerate(rvecs_list):
            x[:, j] = rvecs[:, 0]
            y[:, j] = rvecs[:, 1]
        return x, y

    def filter_correspondences(self, pairs, filter_min=5e-4):
        """
        Filter correspondence pairs to remove redundant or invalid matches.

        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        filter_min : float
            Minimum difference between consecutive points for filtering

        Returns:
        --------
        np.ndarray
            Filtered correspondence pairs
        """
        return self.filter_results(pairs, filter_min)

    def fill_correspondence_gaps(self, pairs, threshold=1e-3, resolution=0.01):
        """
        Fill gaps in the correspondence pairs by computing additional matches.

        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs
        threshold : float
            Maximum chord length for valid matches
        resolution : float
            Minimum resolution between points

        Returns:
        --------
        np.ndarray
            Gap-filled correspondence pairs
        """
        pairs_filled = self.fill_gaps_with_matching(pairs, resolution, threshold)

        xi_gaps = self.find_gaps_xi(pairs_filled, resolution)
        pairs_filled = self.update_xi(pairs_filled.tolist(), xi_gaps, threshold)

        z_gaps = self.find_gaps_z(pairs_filled, resolution)
        pairs_filled = self.update_z(pairs_filled.tolist(), z_gaps, threshold)

        return np.array(sorted(pairs_filled, key=lambda x: x[0]))

    def filter_one_curve(self, pairs, min_diff=5e-4, param='xi', bounds=None):
        """
        Filter pairs based on one parameter ('xi' or 'z'), optionally within bounds of that parameter.

        Parameters:
        -----------
        pairs : np.ndarray
            Array of [xi, z] pairs.
        min_diff : float
            Minimum difference required to keep a point.
        param : str
            'xi' or 'z' — the parameter to filter by.
        bounds : tuple or None
            (min_val, max_val) — only apply filtering where param is within this range.

        Returns:
        --------
        np.ndarray
            Filtered pairs.
        """
        index = 0 if param == 'xi' else 1

        if bounds is not None:
            min_val, max_val = bounds
            mask = (pairs[:, index] >= min_val) & (pairs[:, index] <= max_val)
            inside = pairs[mask]
            outside = pairs[~mask]
        else:
            inside = pairs
            outside = np.empty((0, 2))

        # Apply filtering only to the "inside" region
        if len(inside) == 0:
            return pairs

        filtered = [inside[0]]
        for curr in inside[1:]:
            prev = filtered[-1]
            if curr[index] > prev[index] and abs(curr[index] - prev[index]) > min_diff:
                filtered.append(curr)
        filtered = np.array(filtered)

        # Combine and sort by xi
        combined = np.vstack([outside, filtered])
        return combined[np.argsort(combined[:, 0])]


    @staticmethod
    def centripetal_catmull_rom_spline(x, y, num_points=500, alpha=0.5):
        """
        Compute a centripetal Catmull-Rom spline interpolation.

        Parameters:
        -----------
        x, y : np.ndarray
            Original coordinates of the points.
        num_points : int
            Number of points in the interpolated spline.
        alpha : float
            Centripetal parameter (0.5 is recommended for centripetal spline).

        Returns:
        --------
        cs_x, cs_y : np.ndarray
            Interpolated spline points.
        """
        # Compute centripetal parameterization
        points = np.array([x, y]).T
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        t = np.zeros(len(points))
        t[1:] = np.cumsum(distances**alpha)
        t /= t[-1]

        # Interpolate x and y independently using cubic interpolation
        fx = interp1d(t, x, kind='cubic')
        fy = interp1d(t, y, kind='cubic')

        t_new = np.linspace(0, 1, num_points)
        cs_x = fx(t_new)
        cs_y = fy(t_new)

        return cs_x, cs_y

    def to_triangulation(self, x, y):
        """
        Convert structured grid (x, y) to a matplotlib Triangulation object.

        Parameters:
        -----------
        x, y : np.ndarray
            Grid coordinate arrays (shape: ni × nj)

        Returns:
        --------
        matplotlib.tri.Triangulation
        """
        from matplotlib.tri import Triangulation

        ni, nj = x.shape
        points = np.column_stack([x.ravel(), y.ravel()])
        triangles = []

        for i in range(ni - 1):
            for j in range(nj - 1):
                n0 = i * nj + j
                n1 = n0 + 1
                n2 = n0 + nj
                n3 = n2 + 1
                triangles.append([n0, n2, n1])
                triangles.append([n1, n2, n3])

        triangles = np.array(triangles)
        return Triangulation(points[:, 0], points[:, 1], triangles)


# EOF
