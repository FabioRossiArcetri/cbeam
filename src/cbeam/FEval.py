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
    and fails method dispatch in FEval.jl."""
    a = _np.asarray(a) if dtype is None else _np.asarray(a, dtype=dtype)
    return _np.ascontiguousarray(a)

# ===== ADD THIS: Load the FEval Julia module =====
_cbeam_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_feval_jl = os.path.join(_cbeam_root, "cbeam", "FEval", "src", "FEval.jl")
jl.seval(f'include("{_feval_jl}")')
# ================================================


def create_tree(points,connections):
    return jl.FEval.construct_tritree(points, connections+1)

def create_tree_from_mesh(mesh):
    return jl.FEval.construct_tritree(mesh.points,mesh.cells[1].data+1)

def sort_mesh(mesh):
    mesh.tree = create_tree_from_mesh(mesh)
    return mesh

def query(point,tree):
    jl_idx = jl.FEval.query(point,tree)
    return jl_idx-1

def evaluate(point,field,tree):
    point = _host(point, dtype=_np.float64)
    field = _host(field, dtype=_np.float64)
    if point.ndim == 2:
        return xp.array(jl.FEval.evaluate(point[:,:2], field, tree))
    return xp.array(jl.FEval.evaluate(point, field, tree))

def resample(field, mesh, newmesh):
    tree = create_tree_from_mesh(mesh)
    return evaluate(newmesh.points, field, tree)

def evaluate_grid(pointsx, pointsy, field, tree):
    return jl.FEval.evaluate(pointsx, pointsy, field, tree)

def update_tree(tree, rescale_factor):
    jl.FEval.update_tritree(tree, rescale_factor)

def evaluate_func(field, tree):
    return jl.FEval.evaluate_func(field, tree)

def transverse_gradient(field, tris, points):
    field  = _host(field, dtype=_np.float64)
    tris   = _host(tris)
    points = _host(points, dtype=_np.float64)
    return xp.array(jl.FEval.transverse_gradient(field, tris, points))

def get_triangles(mesh):
    return mesh.cells[1].data

def get_points(mesh):
    return mesh.points