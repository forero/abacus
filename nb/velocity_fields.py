import abacus_cosmos.Halos as ach
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import Counter
import scipy.ndimage as scpimg

def smooth_data(path, L_cell=10.0, sigma_cut=300.0, sigma_smooth=1.0, L_box=1100.0):
    output_path = os.path.join(path, "fields")
    halo_data = ach.read_halos_FoF(path)
    halo_table = Table(halo_data)
    sigma_v = halo_data['vcirc_max']
    print("Done reading data")

    N_side = np.int(L_box/L_cell)

    ii = (sigma_v>sigma_cut)
    pos_cut = halo_data['pos'][ii]
    vel_cut = halo_data['vel'][ii]
    min_pos = -L_box/2.0

    print(L_box, L_cell, N_side, len(pos_cut))
    print("Done selection by sigma")

    # Set the coordinate origin to 0.0
    pos_cut[:,0] = pos_cut[:,0] - min_pos
    pos_cut[:,1] = pos_cut[:,1] - min_pos
    pos_cut[:,2] = pos_cut[:,2] - min_pos
    #Nearest Grid Point interpolation
    ii = np.int_(pos_cut[:,0]/L_cell)
    jj = np.int_(pos_cut[:,1]/L_cell)
    kk = np.int_(pos_cut[:,2]/L_cell)

    print(np.min(ii), np.max(ii))
    print(np.min(jj), np.max(jj))
    print(np.min(kk), np.max(kk))

    n_grid = np.zeros((N_side, N_side, N_side))

    vel_x_grid = np.zeros((N_side, N_side, N_side))
    vel_x = vel_cut[:,0]
    for i,j,k,t in zip(ii,jj,kk,range(len(vel_x))):
        vel_x_grid[i,j,k] += vel_x[t]
        n_grid[i,j,k] += 1

    vel_y_grid = np.zeros((N_side, N_side, N_side))
    vel_y = vel_cut[:,1]
    for i,j,k,t in zip(ii,jj,kk,range(len(vel_y))):
        vel_y_grid[i,j,k] += vel_y[t]

    vel_z_grid = np.zeros((N_side, N_side, N_side))
    vel_z = vel_cut[:,2]
    for i,j,k,t in zip(ii,jj,kk,range(len(vel_z))):
        vel_z_grid[i,j,k] += vel_z[t]
    
    zz = n_grid>0
    vel_x_grid[zz] = vel_x_grid[zz]/n_grid[zz]
    vel_y_grid[zz] = vel_y_grid[zz]/n_grid[zz]
    vel_z_grid[zz] = vel_z_grid[zz]/n_grid[zz]

    print("Done NGP interpolation")

    field = "vel_x_NGP"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell)
    np.save(os.path.join(output_path, filename), vel_x_grid)
    field = "vel_y_NGP"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell)
    np.save(os.path.join(output_path, filename), vel_y_grid)
    field = "vel_z_NGP"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell)
    np.save(os.path.join(output_path, filename), vel_z_grid)
    print("Done writing NGP velocities")

    vel_x_grid_smooth = scpimg.filters.gaussian_filter(vel_x_grid,sigma_smooth)
    vel_y_grid_smooth = scpimg.filters.gaussian_filter(vel_y_grid,sigma_smooth)
    vel_z_grid_smooth = scpimg.filters.gaussian_filter(vel_z_grid,sigma_smooth)
    print("Done Gaussian Smoothing")


    field = "vel_x_gauss"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}_smooth_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell, sigma_smooth)
    np.save(os.path.join(output_path, filename), vel_x_grid_smooth)
    field = "vel_y_gauss"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}_smooth_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell, sigma_smooth)
    np.save(os.path.join(output_path, filename), vel_y_grid_smooth)
    field = "vel_z_gauss"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}_smooth_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell, sigma_smooth)
    np.save(os.path.join(output_path, filename), vel_z_grid_smooth)

    print("Done writing gaussian velocities")


    div_x = np.zeros(np.shape(vel_x_grid_smooth))
    div_x[1:-1,:,:] = (vel_x_grid_smooth[2:,:,:] - vel_x_grid_smooth[:-2,:,:])/(2.0*L_cell)
    div_x[0,:,:] = (vel_x_grid_smooth[1,:,:] - vel_x_grid_smooth[-1,:,:])/(2.0*L_cell)
    div_x[-1,:,:] = (vel_x_grid_smooth[0,:,:] - vel_x_grid_smooth[-2,:,:])/(2.0*L_cell)

    div_y = np.zeros(np.shape(vel_y_grid_smooth))
    div_y[:,1:-1,:] = (vel_y_grid_smooth[:,2:,:] - vel_y_grid_smooth[:,:-2,:])/(2.0*L_cell)
    div_y[:,0,:] = (vel_y_grid_smooth[:,1,:] - vel_y_grid_smooth[:,-1,:])/(2.0*L_cell)
    div_y[:,-1,:] = (vel_y_grid_smooth[:,0,:] - vel_y_grid_smooth[:,-2,:])/(2.0*L_cell)
    
    div_z = np.zeros(np.shape(vel_z_grid_smooth))
    div_z[:,:,1:-1] = (vel_z_grid_smooth[:,:,2:] - vel_z_grid_smooth[:,:,:-2])/(2.0*L_cell)
    div_z[:,:,0] = (vel_z_grid_smooth[:,:,1] - vel_z_grid_smooth[:,:,-1])/(2.0*L_cell)
    div_z[:,:,-1] = (vel_z_grid_smooth[:,:,0] - vel_z_grid_smooth[:,:,-2])/(2.0*L_cell)

    div = div_x + div_y + div_z

    print("Done Divergence Computation")

    field = "div"
    filename = "{}_box_{:.1f}_sigmacut_{:.1f}_cell_{:.1f}_smooth_{:.1f}.npy".format(field, L_box, sigma_cut, L_cell, sigma_smooth)
    np.save(os.path.join(output_path, filename), div)
    print("Done writing divergence")

    
full_computation = True

if full_computation:
    path = "../data/AbacusCosmos_1100box_00_FoF_halos_z0.300/"
    for L_cell, sigma_smooth in zip([10.0, 5.0, 2.5], [1.0, 2.0, 4.0]):
        for sigma_cut in [200.0, 300.0, 400.0]:
            smooth_data(path, L_cell=L_cell, sigma_cut=sigma_cut, sigma_smooth=sigma_smooth)