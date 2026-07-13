from juliacall import Main as jl
from .backend import get_xp
import os, cbeam

from juliacall import Main as jl

# Load Pkg natively using Julia syntax
jl.seval("using Pkg")
Pkg = jl.Pkg
xp = get_xp()

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
    return xp.array(jl.FEval.transverse_gradient(field, tris, points))

def get_triangles(mesh):
    return mesh.cells[1].data

def get_points(mesh):
    return mesh.points