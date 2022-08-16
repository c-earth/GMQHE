#############################
# main file for running GMQHE
#############################

# import modules
from turtle import position
import kwant
import numpy as np
import tinyarray as ta
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy
from utils.data_prep import TB_model
from utils.build import create_specimens, lead_attaching, create_platform, etching

# work flow settings
regenerate_tb_models = True
rebuild_specimens = True

# paths settings
raw_data_folder = './data/raw'
tb_models_folder = './data/TB_models'
specimens_folder = './data/specimens'

# load data
TB_models = TB_model.from_folder(raw_data_folder)

# # specify orders
# orders = {}
# lead_orders = {}

# # build specimens
# specimens = create_specimens(orders)

# # attach leads
# specimens = lead_attaching(specimens, lead_orders)
x_size = 20
y_size = 20

def rectangle(pos):
    x, y = pos
    return abs(x) < x_size and abs(y) < y_size

def rectangle_hole(pos):
    x, y = pos
    return -x_size/4 <x < x_size/2 and -y_size/4 <y < y_size/2
lattice = kwant.lattice.Monatomic(TB_models['1Tprime_MoS2'].structure.lattice.matrix[:2, :2])
specimen = create_platform(TB_models['1Tprime_MoS2'], rectangle, (0, 0), 3)
# specimen = etching(specimen, TB_models['1Tprime_MoS2'], rectangle_hole, (0, 0))

# wrapped = kwant.wraparound.wraparound(specimen).finalized()
# # kwant.wraparound.plot_2d_bands(wrapped)

# def spectrum(syst, x, y=None, params=None, mask=None, file=None,
#              show=True, dpi=None, fig_size=None, ax=None):
#     _p = kwant._common.lazy_import('_plotter')
#     if not _p.mpl_available:
#         raise RuntimeError("matplotlib was not found, but is required "
#                            "for plot_spectrum()")
#     if y is not None and not _p.has3d:
#         raise RuntimeError("Installed matplotlib does not support 3d plotting")

#     if isinstance(syst, kwant.system.FiniteSystem):
#         def ham(**kwargs):
#             return syst.hamiltonian_submatrix(params=kwargs, sparse=False)
#     elif callable(syst):
#         ham = syst
#     else:
#         raise TypeError("Expected 'syst' to be a finite Kwant system "
#                         "or a function.")

#     params = params or dict()
#     keys = (x[0],) if y is None else (x[0], y[0])
#     array_values = (x[1],) if y is None else (x[1], y[1])
#     import functools, itertools
#     # calculate spectrum on the grid of points
#     spectrum = []
#     bound_ham = functools.partial(ham, **params)
#     for point in itertools.product(*array_values):
#         p = dict(zip(keys, point))
#         if mask and mask(**p):
#             spectrum.append(None)
#         else:
#             h_p = np.atleast_2d(bound_ham(**p))
#             spectrum.append(np.linalg.eigvalsh(h_p))
#     # massage masked grid points into a list of NaNs of the appropriate length
#     shape_eigvals = next(filter(lambda s: s is not None, spectrum)).shape
#     nan_list = np.full(shape_eigvals, np.nan)
#     spectrum = [nan_list if s is None else s for s in spectrum]
#     # make into a numpy array and reshape
#     new_shape = [len(v) for v in array_values] + [-1]
#     spectrum = np.array(spectrum).reshape(new_shape)
#     import warnings
#     # set up axes
#     if ax is None:
#         fig = kwant.plotter._make_figure(dpi, fig_size, use_pyplot=(file is None))
#         # if y is None:
#         ax = fig.add_subplot(1, 1, 1)
#         # else:
#         #     warnings.filterwarnings('ignore',
#         #                             message=r'.*mouse rotation disabled.*')
#         #     ax = fig.add_subplot(1, 1, 1, projection='3d')
#         #     warnings.resetwarnings()
#         ax.set_xlabel(keys[0])
#         # if y is None:
#         ax.set_ylabel('Energy')
#         # else:
#         #     ax.set_ylabel(keys[1])
#         #     ax.set_zlabel('Energy')
#         # ax.set_title(
#         #     ', '.join(
#         #         '{} = {}'.format(key, value)
#         #         for key, value in params.items()
#         #         if not callable(value)
#         #     )
#         # )
#     else:
#         fig = None

#     # actually do the plot
#     # if y is None:
#     ax.plot(array_values[0], spectrum[:, 16])
#     # else:
#     #     if not hasattr(ax, 'plot_surface'):
#     #         msg = ("When providing an axis for plotting over a 2D domain the "
#     #                "axis should be created with 'projection=\"3d\"")
#     #         raise TypeError(msg)
#     #     # plot_surface cannot directly handle rank-3 values, so we
#     #     # explicitly loop over the last axis
#     #     grid = np.meshgrid(*array_values)
#     #     with warnings.catch_warnings():
#     #         warnings.filterwarnings('ignore', message='Z contains NaN values')
#     #         for i in range(spectrum.shape[-1]):
#     #             spec = spectrum[:, :, i].transpose()  # row-major to x-y ordering
#     #             ax.plot_surface(*(grid + [spec]), cstride=1, rstride=1)

#     kwant.plotter._maybe_output_fig(fig, file=file, show=show)

#     return fig


# def plot_unwrapped_2d_bands(syst, k_x=31, k_y=31, params=None,
#                   mask_brillouin_zone=False, extend_bbox=0, file=None,
#                   show=True, dpi=None, fig_size=None, ax=None):
#     if not hasattr(syst, '_wrapped_symmetry'):
#         raise TypeError("Expecting a system that was produced by "
#                         "'kwant.wraparound.wraparound'.")
#     if isinstance(syst, kwant.system.InfiniteSystem):
#         msg = ("All symmetry directions must be wrapped around: specify "
#                "'keep=None' when calling 'kwant.wraparound.wraparound'.")
#         raise TypeError(msg)
#     if isinstance(syst, kwant.builder.Builder):
#         msg = ("Expecting a finalized system: remember to finalize your "
#                "system with 'syst.finalized()'.")
#         raise TypeError(msg)

#     params = params or {}
#     lat_ndim, space_ndim = syst._wrapped_symmetry.periods.shape

#     if lat_ndim != 2:
#         raise ValueError("Expected a system with a 2D translational symmetry.")
#     if space_ndim != lat_ndim:
#         raise ValueError("Lattice dimension must equal realspace dimension.")

#     # columns of B are lattice vectors
#     B = np.array(syst._wrapped_symmetry.periods).T
#     # columns of A are reciprocal lattice vectors
#     A = np.linalg.pinv(B).T

#     ## calculate the bounding box for the 1st Brillouin zone
#     from kwant.linalg import lll
#     # Get lattice points that neighbor the origin, in basis of lattice vectors
#     reduced_vecs, transf = lll.lll(A.T)
#     neighbors = ta.dot(lll.voronoi(reduced_vecs), transf)
#     # Add the origin to these points.
#     klat_points = np.concatenate(([[0] * lat_ndim], neighbors))
#     # Transform to cartesian coordinates and rescale.
#     # Will be used in 'outside_bz' function, later on.
#     klat_points = 2 * np.pi * np.dot(klat_points, A.T)
#     # Calculate the Voronoi cell vertices
#     vor = scipy.spatial.Voronoi(klat_points)
#     around_origin = vor.point_region[0]
#     bz_vertices = vor.vertices[vor.regions[around_origin]]
#     # extract bounding box
#     k_max = np.max(np.abs(bz_vertices), axis=0)

#     ## build grid along each axis, if needed
#     ks = []
#     for k, km in zip((k_x, k_y), k_max):
#         k = np.array(k)
#         if not k.shape:
#             if extend_bbox:
#                 km += km * extend_bbox
#             k = np.linspace(-km, km, k)
#         ks.append(k)

#     # TODO: It is very inefficient to call 'momentum_to_lattice' once for
#     #       each point (for trivial Hamiltonians 60% of the time is spent
#     #       doing this). We should instead transform the whole grid in one call.

#     def momentum_to_lattice(k):
#         k, residuals = scipy.linalg.lstsq(A, k)[:2]
#         if np.any(abs(residuals) > 1e-7):
#             raise RuntimeError("Requested momentum doesn't correspond"
#                                " to any lattice momentum.")
#         return k

#     def ham(k_x, k_y=None, **params):
#         # transform into the basis of reciprocal lattice vectors
#         k = momentum_to_lattice([k_x] if k_y is None else [k_x, k_y])
#         p = dict(zip(syst._momentum_names, k), **params)
#         return syst.hamiltonian_submatrix(params=p, sparse=False)

#     def outside_bz(k_x, k_y, **_):
#         dm = scipy.spatial.distance_matrix(klat_points, [[k_x, k_y]])
#         return np.argmin(dm) != 0  # is origin no closest 'klat_point' to 'k'?

#     fig = spectrum(ham,
#                            x=('k_x', ks[0]),
#                            y=('k_y', ks[1]) if lat_ndim == 2 else None,
#                            params=params,
#                            mask=(outside_bz if mask_brillouin_zone else None),
#                            file=file, show=show, dpi=dpi,
#                            fig_size=fig_size, ax=ax)
#     return fig


# plot_unwrapped_2d_bands(wrapped)

specimen =  specimen.finalized()

# construct the Haldane model
lat, fsyst = lattice, specimen
# find 'A' and 'B' sites in the unit cell at the center of the disk
# where = lambda s: np.linalg.norm(s.pos) < 1
# component 'xx'
# s_factory = kwant.kpm.LocalVectors(specimen, where)
print('s')
cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')
# component 'xy'
print(0)
# s_factory = kwant.kpm.LocalVectors(fsyst, where)
cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')
energies = cond_xx.energies
# print(energies)
cond_array_xx = np.array([cond_xx(e, temperature=0.1) for e in energies])
print(1)
cond_array_xy = np.array([cond_xy(e, temperature=0.1) for e in energies])
# area of the unit cell per site
area_per_site = np.abs(np.cross(*lat.prim_vecs) / len(lat.sublattices))
cond_array_xx /= area_per_site
cond_array_xy /= area_per_site
plt.figure()
plt.plot(energies, cond_array_xx, label = 'xx')
plt.plot(energies, cond_array_xy, label = 'xy')
plt.legend()
plt.show()