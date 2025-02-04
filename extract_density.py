# Wzhi: Extract 2d densities from drawings
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import copy
import pickle

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn

from torchvision.utils import save_image

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
max_y = 600  # image_size_max

n_images = 2  # we have 2 images
n_trajs = 3  # we have 3 trajectories
traj_len_est = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_traj(view_str="bullet_view_{}.png", ts=20):
    global img
    list_xy = []

    # define mouse callback function to draw circle
    def draw_curve(event, x, y, flags, param):
        global ix, iy, drawing, img
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing == False:
                drawing = True
            else:
                drawing = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                list_xy.append([x, y])
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    traj_all_list = []
    for ii in range(0, n_images):
        traj_list = []
        for jj in range(n_trajs):
            list_xy = []
            img = cv2.imread(view_str.format(ii))

            cv2.namedWindow("Curve Window")
            cv2.setMouseCallback("Curve Window", draw_curve)

            while True:
                cv2.imshow("Curve Window", img)
                cv2.moveWindow("Curve Window", 100, 100)
                if cv2.waitKey(10) == 27:
                    break
            cv2.destroyAllWindows()
            xy_tor = torch.tensor(list_xy)
            # print(xy_tor.shape)
            xy_tor[:, 1] = max_y - xy_tor[:, 1]
            traj_list.append(xy_tor.detach().clone())
        traj_all_list.append(copy.deepcopy(traj_list))
    grand_traj_l = []

    for im in range(n_images):
        traj_list_s = []
        plt.figure()
        for i in range(n_trajs):
            # traj_len = int(len(traj_all_list[im][i]) / (ts))
            # traj_sketch = traj_all_list[im][i][::traj_len, :][:ts]
            # traj_sketch_end = traj_all_list[im][i][-traj_len_est:-1, :]
            # traj_all = torch.vstack([traj_sketch, traj_sketch_end])
            traj_all = traj_all_list[im][i] / max_y
            plt.scatter(traj_all[:, 0], traj_all[:, 1])
            # print(len(traj_all))
            traj_list_s.append(traj_all[None])
        # traj_tor = traj_all  # torch.vstack(traj_list_s)[None]
        grand_traj_l.append(copy.deepcopy(traj_list_s))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.savefig("bullet_traj_table{}.png".format(im))

    # normalise to 1
    # grand_traj_tor = torch.vstack(grand_traj_l)
    # grand_traj_tor_r = grand_traj_tor / max_y
    return grand_traj_l


def generate_densities(
    grand_traj_tor_r, noise_added=0.001, img_len=100, time_length=50, hdim=512
):
    N_DIM = 3

    # create grid to query

    xy_g = torch.linspace(0, 1, img_len)
    grid_xy_vals = torch.meshgrid(xy_g, xy_g)
    grid_xy = torch.cat(
        [grid_xy_vals[0][:, :, None], grid_xy_vals[1][:, :, None]], dim=-1
    ).reshape((-1, 2))

    # number of times steps of images produced
    time_range = torch.linspace(0, 1, time_length)

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, hdim), nn.ReLU(), nn.Linear(hdim, dims_out)
        )

    for view in range(n_images):
        t_view_wt = grand_traj_tor_r[view]
        # time_len = t_view.shape[1]
        # times_tor = (torch.arange(0, time_len) / time_len)[None, :, None]
        # times_tor_tile = times_tor.repeat((t_view.shape[0], 1, 1))
        # t_view_wt = torch.cat([times_tor_tile, t_view], dim=-1)
        data = t_view_wt.reshape((-1, 3)).to(device)
        inn = Ff.SequenceINN(N_DIM)
        for k in range(8):
            inn.append(
                Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True
            )
        inn = inn.to(device)
        optimizer = torch.optim.Adam(inn.parameters(), lr=0.0003, weight_decay=0.00005)

        for i in range(5000):
            optimizer.zero_grad()
            noise_in_space = (torch.randn_like(data) * noise_added).to(device)
            x = (data + noise_in_space).to(device)
            # pass to INN and get transformed variable z and log Jacobian determinant
            z, log_jac_det = inn(x)
            # calculate the negative log-likelihood of the model with a standard normal prior
            loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean() / N_DIM
            # backpropagate and update the weights
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print("{}: {}".format(i, loss))

        # create densities along time and save to a folder called "n_imgs"
        print("creating densities for view {}".format(view))
        for times_ind in range(len(time_range)):
            query_txy = torch.cat(
                [time_range[times_ind] * torch.ones(len(grid_xy), 1), grid_xy], dim=-1
            ).to(device)
            z, log_jac_det = inn(query_txy)
            loss = torch.exp(-0.5 * torch.sum(z**2, 1) - log_jac_det)
            loss_r = loss / loss.max()
            grid_xy_im = loss_r.reshape((img_len, img_len)).swapaxes(0, 1).cpu()
            # convert to img
            img = torch.zeros((img_len, img_len))
            for i in range(grid_xy_im.shape[0]):
                for j in range(grid_xy_im.shape[1]):
                    img[img_len - 1 - i, j] = grid_xy_im[i, j]
            save_image(img.detach(), "traj_imgs/img_{}_{}.png".format(view, times_ind))
        inn = inn.cpu()
