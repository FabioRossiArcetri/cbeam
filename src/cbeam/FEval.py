from juliacall import Main as jl
from .backend import get_xp
import os, cbeam
import numpy as _np

from juliacall import Main as jl

# Load Pkg natively using Julia syntax
jl.seval("using Pkg")
Pkg = jl.Pkg
xp = get_xp()


def _host(a, dtype=None):
    """Coerce an array (possibly a JAX device array) to a contiguous host numpy
    array before handing it to Julia.  PythonCall wraps a numpy ndarray as a
    zero-copy ``PyArray``; a JAX array instead arrives as ``PyIterable{Any}``
    and fails method dispatch in FEval.jl.

    With ``dtype=None`` the value is promoted to the dtype FEval.jl expects:
    float64 for real input, complex128 for complex input (its ``evaluate``
    methods are ``T<:Union{Float64,ComplexF64}``); integer/bool arrays such as
    triangle-connectivity are left as they are.
    """
    a = _np.asarray(a)
    if dtype is None:
        if _np.iscomplexobj(a):
            dtype = _np.complex128
        elif _np.issubdtype(a.dtype, _np.floating):
            dtype = _np.float64
    return _np.ascontiguousarray(a, dtype=dtype)

# ===== ADD THIS: Load the FEval Julia module =====
_cbeam_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_feval_jl = os.path.join(_cbeam_root, "cbeam", "FEval", "src", "FEval.jl")
jl.seval(f'include("{_feval_jl}")')
# ================================================


def create_tree(points,connections):
    return jl.FEval.construct_tritree(_host(points, dtype=_np.float64),
                                      _host(connections) + 1)

def create_tree_from_mesh(mesh):
    return jl.FEval.construct_tritree(mesh.points,mesh.cells[1].data+1)

def sort_mesh(mesh):
    mesh.tree = create_tree_from_mesh(mesh)
    return mesh

def query(point,tree):
    jl_idx = jl.FEval.query(_host(point, dtype=_np.float64), tree)
    return jl_idx-1

def evaluate(point,field,tree):
    point = _host(point, dtype=_np.float64)
    field = _host(field)                       # float64 or complex128
    if point.ndim == 2:
        return xp.array(jl.FEval.evaluate(point[:,:2], field, tree))
    return xp.array(jl.FEval.evaluate(point, field, tree))

def resample(field, mesh, newmesh):
    tree = create_tree_from_mesh(mesh)
    return evaluate(newmesh.points, field, tree)

def evaluate_grid(pointsx, pointsy, field, tree):
    pointsx = _host(pointsx, dtype=_np.float64)
    pointsy = _host(pointsy, dtype=_np.float64)
    field   = _host(field)                     # float64 or complex128
    return jl.FEval.evaluate(pointsx, pointsy, field, tree)

def update_tree(tree, rescale_factor):
    jl.FEval.update_tritree(tree, rescale_factor)

def evaluate_func(field, tree):
    return jl.FEval.evaluate_func(field, tree)

def transverse_gradient(field, tris, points):
    field  = _host(field)                      # float64 or complex128
    tris   = _host(tris)
    points = _host(points, dtype=_np.float64)
    return xp.array(jl.FEval.transverse_gradient(field, tris, points))

def get_triangles(mesh):
    return mesh.cells[1].data

def get_points(mesh):
    return mesh.points