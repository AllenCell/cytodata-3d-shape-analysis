import numpy as np
import pandas as pd

import pyvista as pv
import vtk
from vtk.util import numpy_support

from aicsimageio.writers import OmeTiffWriter
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools


def approximate_one_param(z, thresh):
    n = len(z)
    zhat = np.fft.fft(z, n)
    PSD = np.real(zhat * np.conj(zhat)/n)
    indices = PSD > thresh
    PSDclean = PSD * indices
    zhat = indices * zhat
    zfilt = np.fft.ifft(zhat)
    return zfilt

def get_mesh_from_series(coeff_dict, lmax):
    row = pd.Series(coeff_dict)
    coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
    for l in range(lmax):
        for m in range(l + 1):
            try:
                # Cosine SHE coefficients
                coeffs[0, l, m] = row[
                    [f for f in row.keys() if f"shcoeffs_L{l}M{m}C" in f]
                ]
                # Sine SHE coefficients
                coeffs[1, l, m] = row[
                    [f for f in row.keys() if f"shcoeffs_L{l}M{m}S" in f]
                ]
            # If a given (l,m) pair is not found, it is assumed to be zero
            except: pass
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
    return mesh

def vtk_polydata_to_imagedata(polydata, dimensions=(64,64,64), padding=0):
    xi, xf, yi, yf, zi, zf = polydata.GetBounds()
    dx, dy, dz = dimensions
    sx = (xf - xi) / dx
    sy = (yf - yi) / dy
    sz = (zf - zi) / dz
    ox = xi + sx / 2.0
    oy = yi + sy / 2.0
    oz = zi + sz / 2.0

    if padding:
        ox -= sx
        oy -= sy
        oz -= sz

        dx += 2 * padding
        dy += 2 * padding
        dz += 2 * padding

    image = vtk.vtkImageData()
    image.SetSpacing((sx, sy, sz))
    image.SetDimensions((dx, dy, dz))
    image.SetExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
    image.SetOrigin((ox, oy, oz))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    inval = 255
    outval = 0

    for i in range(image.GetNumberOfPoints()):
        image.GetPointData().GetScalars().SetTuple1(i, inval)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin((ox, oy, oz))
    pol2stenc.SetOutputSpacing((sx, sy, sz))
    pol2stenc.SetOutputWholeExtent(image.GetExtent())
    pol2stenc.Update()
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc.GetOutput()

def vtk_image_to_numpy_image(vtk_image):
    dims = vtk_image.GetDimensions()
    data = vtk_image.GetPointData().GetScalars()
    np_image = numpy_support.vtk_to_numpy(data)
    np_image = np_image.reshape(dims, order='F')
    return np_image

def get_image_from_polydata(mesh):
    vtk_image = vtk_polydata_to_imagedata(mesh)
    np_image = vtk_image_to_numpy_image(vtk_image)
    return np_image

# GIF method for viz reconstruction error
# import glob
# mesh_files = sorted(glob.glob("output/recon-0*.vtk"))

# plotter = pv.Plotter(notebook=False, off_screen=True)
# plotter.open_gif("output/lmax_reconstruction_nucleus.gif")
# plotter.set_background("white")

# recon_mesh = pv.read(mesh_files[10])
# meshes = gt_mesh + recon_mesh
# meshes = meshes.rotate_z(-20, inplace=True)
# plotter.add_mesh(meshes, show_scalar_bar=False)
# l = plotter.add_text(f"Lmax=1 Reconstruction error: {recon_errors[0]:.2f}", color="black")

# for i,f in enumerate(mesh_files[1:]):
#     recon_mesh = pv.read(f)
#     meshes.overwrite(gt_mesh+recon_mesh)
#     _ = plotter.remove_actor(l)
#     plotter.add_mesh(meshes, show_scalar_bar=False)
#     l = plotter.add_text(f"Lmax={i+2} Reconstruction error: {recon_errors[i]:.2f}", color="black")
#     plotter.render()
#     plotter.write_frame()
#     plotter.write_frame()
#     plotter.write_frame()

# plotter.close()

# from IPython.display import Image
# display(Image(data=open("output/lmax_reconstruction_nucleus.gif","rb").read(), format="png"))