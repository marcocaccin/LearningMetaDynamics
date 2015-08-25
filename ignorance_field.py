import scipy as sp
import scipy.linalg as LA
import scipy.spatial.distance as spdist
import warnings 

def v2min_image_v(dr, cell, pbc=None, shifts_out=False):
    """
    ------
    Authors: matscipy authors - https://github.com/libAtoms/matscipy

    ------
    Apply minimum image convention to an array of distance vectors.

    Parameters
    ----------
    dr : array_like
        Array of distance vectors.
    cell : array_like, shape (n_dim,)
        Cell extent in each direction.
    pbc : array_like, optional, type bool
        Periodic boundary conditions directions. Default is to
        assume periodic boundaries in all directions.

    Returns
    -------
    dr : array
        Array of distance vectors, wrapped according to the minimum image
        convention.

    """
    # Check where distance larger than 1/2 cell. Particles have crossed
    # periodic boundaries then and need to be unwrapped.
    n_dim = len(cell)
    rec = sp.diag(1. / sp.asarray(cell))
    cell = sp.diag(sp.asarray(cell))
    if pbc is not None:
        rec *= sp.array(pbc, dtype=int).reshape(n_dim, 1)
    dri = sp.round_(sp.dot(dr, rec))
    shifts = sp.dot(dri, cell)
    # Unwrap
    if shifts_out:
        return dr - shifts, shifts
    else:
        return dr - shifts


def round_vector(vec, precision = 0.05):
    """
    Rounds an array with the required precision.
    """
    return ((vec + 0.5 * precision) / precision).astype('int') * precision


# def unique_vectors(v):
#     """
#     Unique vectors of a list of vectors.
#     """
#     vstr = [str(x) for x in v]
#     unique_vstr, unique_idx = sp.unique(vstr, return_index = True)
#     unique_v = v[unique_idx]
#     return unique_v


def where_a_in_b(a, b):
    a, b = sp.atleast_2d(a), sp.atleast_2d(b)
    indices = spdist.cdist(a, b)
    indices = sp.where(indices < 1.0e-8)[1] # get indices of array b
    return indices


class IgnoranceField:
    def __init__(self, X_grid, y_threshold=1.0e-1, **kwargs):
        """
        Parameters:
        ----------
        X_grid: array_like, shape (n_samples, n_features)
            Array of points that tessellate a patch of space
        y_threshold: real
            Value above which a "cost wall" is detected
        
        kwargs:
        ----------
        cell : array_like, shape (n_features,)
            Cell extent in each direction.
        pbc : array_like, type bool
            Periodic boundary conditions directions. Default is to
            assume periodic boundaries in all directions.
        X_grid_spacing: real
            Grid spacing between points in X
        boundaries: array_like, shape(n_features,)
            max - min in each direction of X
        cutoff: real, > 0
            distance within which points are interacting with current point.
        """

        self.X_grid = X_grid # 2D array: [[x0min, x0max],...,[xNmin, xNmax]] 
        self.y_threshold = y_threshold

        self.pbc = kwargs.get('pbc', None)
        # spacing between grid points is the minimum nonzero distance between them.
        spacing_inferred = sp.sort(spdist.cdist(X_grid, sp.atleast_2d(X_grid[0])))[1].item()
        self.X_grid_spacing = kwargs.get('X_grid_spacing', spacing_inferred)
        # 
        boundaries_inferred = sp.array([x.max() - x.min() for x in X_grid.T])
        self.boundaries = kwargs.get('boundaries', boundaries_inferred)
        self.cutoff = kwargs.get('cutoff', None)
        self.y_grid = self.y_threshold * sp.ones(len(X_grid)) - 0.1
        self.n_grid = sp.ones(len(X_grid))
        self.wall = sp.zeros(len(X_grid), dtype=bool)


    def set_cost_grid(self, y_grid, n_grid=None):
        """
        On a given grid in the X plane, set the values in y and 
        the ignorance for each value.

        Parameters:
        ----------
        y_grid: array_like, shape (n_samples,)
            Array of scalar-valued function on the grid points X_grid that
            determines the "cost" of each point. If cost > threshold, the point forms a wall
        n_grid: array_like, shape (n_samples,)
            Array of scalar-valued function on the grid points X_grid that
            sets the ignorance level for each point. Higher values attract more.
            
        Other properties:
        ----------
        wall: array_like, shape (n_samples,), type boolean
           Denotes the presence or not of a wall for each point in X_grid.
        """

        assert len(self.X_grid) == len(y_grid)
        if n_grid is None or y_grid.shape != n_grid.shape:
            n_grid = sp.ones(y_grid.shape)
        self.y_grid = y_grid.flatten()
        self.n_grid = n_grid
        self.wall = (y_grid > self.y_threshold)


    def update_cost(self, X, y):
        """
        For an ignorance function 1/N, where N is number of times
        the system passed on a point X, update ignorance and wall.
        +1 the n_grid
        """
        Xs = sp.atleast_2d(X)
        ys = sp.atleast_1d(y)
        for X, y in zip(Xs, ys):
            X = X.flatten()
            index = where_a_in_b(X, self.X_grid).item()
            self.y_grid[index] = y
            self.wall[index] = (y > self.y_threshold)
            self.n_grid[index] = 1. / (self.n_grid[index] + 1)


    def distance_vectors_not_walled(self, X0):
        """
        Parameters:
        ----------
        X0: array_like, shape (n_features,)
            Point with respect to which the ignorance is calculated.

        Returns:
        ----------
        vectors_mic: array_like, shape (n_vecs, n_features)
            Array of distance vectors connecting point X0 and
            points on a grid self.X_grid which do not cross a
            region for which y > threshold.
        dists: array_like, shape (n_vecs,)
            Euclidean length of vectors in dist_vs
        n_grid: array_like, shape (n_vecs,)
            Ignorance value for each vectors in vectors_mic.
        """

        # distance vectors between current point X0 and all points on a grid,
        # calculated in PBC minimum image convention.
        vectors_mic, shifts = v2min_image_v(
            self.X_grid - X0, self.boundaries, shifts_out=True)
        # norms of distance vectors
        dists = sp.array(map(sp.linalg.norm, vectors_mic))
        if self.cutoff is not None:
            mask = dists <= self.cutoff
            vectors_mic = vectors_mic[mask]
            dists = dists[mask]
            wall = self.wall[mask]
            n_grid = self.n_grid[mask]
        else:
            wall = self.wall
            n_grid = self.n_grid
        # vectors which contain a wall
        X_wall = vectors_mic[wall]
        # distances between each wall point and X0
        X_wall_dists = sp.array(map(LA.norm, X_wall))
        # indices of vectors that are in the direction of a wall
        cosines = spdist.cdist(vectors_mic, X_wall, metric='cosine')
        wall_direction = sp.where(cosines < 1.0e-3) # arbitrary tolerance
        # element by element comparison: is point beyond the wall or not?
        beyond_walls = (X_wall_dists[wall_direction[1]] <= dists[wall_direction[0]])
        # indices of points that are in the direction of a wall, and their distances 
        # are larger that the distance of the corresponding wall point.
        walled_indices = wall_direction[0][sp.where(beyond_walls)[0]]
        all_indices = sp.arange(len(vectors_mic))
        no_wall_indices = sp.array(list(set(all_indices) - set(walled_indices)))
        vectors_mic = vectors_mic[no_wall_indices]
        dists = dists[no_wall_indices]
        n_grid = n_grid[no_wall_indices]
        dists[dists < self.X_grid_spacing] = self.X_grid_spacing
        return vectors_mic, dists, n_grid


    def get_forces(self, X0, direction_only=True):
        """
        Parameters:
        ----------
        X0: array_like, shape (n_features,)
            Point with respect to which the ignorance is calculated.

        Returns:
        ----------
        field: array_like, shape (n_features,)
            Force vector pointing towards the direction of maximum ignorance.
        """
        dvecs, dnorms, ignorances = self.distance_vectors_not_walled(X0)
        # weights can be substituted with something less naive than
        # number of times the simulation crossed a point, i.e. MSE
        # Electrostatic-like field: \sum_i q_i / |r_i|^3 * \mathbf{r}_i
        field = ((ignorances / dnorms**3)[:,None] * dvecs).sum(axis=0)
        if direction_only:
            field /= LA.norm(field)
        return field


# here follows an earlier algorithm of distance_vectors_not_walled, 
# which was way too expensive, but too nice to delete.
#
#         # each distance vector is discretised at points that are spaced by the grid spacing
#         discrete_ns = (dists / self.X_grid_spacing).astype('int') + 1
#         # same mesh size and ponts of given X_grid
#         discretised_vectors_mic = [round_vector(
#                 sp.array(
#                     vec * sp.linspace(0, 1, n)[:,None]
#                     ), precision = self.X_grid_spacing) 
#                 for vec, n in zip(vectors_mic, discrete_ns)]
#         no_wall_present = []
#         for dd in discretised_vectors_mic:
#             try:
#                 indices = where_a_in_b(dd, X_grid_mic)
#                 # is there a wall anywhere?
#                 wall_present = wall[indices].any()
#                 no_wall_present.append(not wall_present)
#             except Exception as err:
#                 # It cannot be determined if the segment crosses a wall.
#                 # In this case, assume there is a wall.
#                 warnings.warn("%s. \t Missing information for wall check." % err)
#                 no_wall_present.append(False)
#         # no_wall_present becomes the indices array of where there is no wall
#         no_wall_present = sp.where(no_wall_present)[0]
#         vectors_mic, dists = vectors_mic[no_wall_present], dists[no_wall_present]
#         # no_wall_indices =  sp.arange(len(self.X_grid))[no_wall_present]
#         n_grid =  n_grid[no_wall_present]
       
