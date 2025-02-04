import numpy as np
import torch
from torch import nn
from rendering_helpers import *


# compute Gaussian Kernels
def rbf_kernel(t, t_s, gamma=50):
    feat_dists = torch.exp(-gamma * (torch.cdist(t, t_s, p=2)) ** 2)
    return feat_dists


class trajectory_model:
    def __init__(
        self,
        focal=None,
        height=100,
        width=100,
        near=1,
        far=6,
        n_weights=20,
        n_views=2,
        n_times=50,
        n_samples=150,
        perturb=False,
        gamma=50,
        ray_dist_threshold=0.01,
        density_threshold=0.25,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        if focal == None:
            # default focal from the lego nerf data
            focal = torch.tensor(138.8888789).to(device)
        self.focal = focal
        self.height = height
        self.width = width
        self.near = near
        self.far = far
        self.n_views = n_views
        self.n_times = n_times
        self.n_samples = n_samples
        self.perturb = perturb
        self.ray_dist_threshold = ray_dist_threshold
        self.density_threshold = density_threshold
        self.n_weights = n_weights
        self.gamma = gamma
        self.device = device
        # initialise weights means and weights standard deviation
        self.weights = torch.zeros((n_weights, 3)).to(device)
        self.std_weights = torch.zeros((n_weights, 3)).to(device)
        # set the time inducing points +-0.2 over the boundaries
        self.t_s = torch.linspace(-0.2, 1.2, n_weights).reshape((-1, 1)).to(device)
        # query resolution set as n_times
        self.t_check = torch.linspace(0, 1, n_times).reshape((-1, 1)).to(device)
        # precompute the features
        self.feats = rbf_kernel(self.t_check, self.t_s, self.gamma)

    def extract_mean_std_from_images(self, im_list_tor_all, poses):
        """
        Here we input the images (time,views,width,height),
        along with poses of each view
        """
        mean_l = []
        var_l = []
        # loop through the times and check for intersecting rays
        for ctime in range(self.n_times):
            im_list_ti = im_list_tor_all[ctime]
            rel_points_list = []
            for i_pose in range(self.n_views):
                cur_img = im_list_ti[i_pose]
                target_pose = poses[i_pose].to(self.device)
                rays_o, rays_d = get_rays(
                    self.height, self.width, self.focal, target_pose
                )
                query_points, z_vals = sample_stratified(
                    rays_o,
                    rays_d,
                    self.near,
                    self.far,
                    n_samples=self.n_samples,
                    perturb=self.perturb,
                )
                rel_points = query_points[cur_img > self.density_threshold, :, :]
                rel_points_list.append(rel_points.detach().clone())
            points_0 = rel_points_list[0].reshape((-1, 3))
            points_1 = rel_points_list[1].reshape((-1, 3))
            dists = torch.cdist(points_0, points_1)
            vals_0, inds_0 = torch.min(dists, dim=-1)
            vals_1, inds_1 = torch.min(dists, dim=0)
            inter_points_0 = points_0[vals_0 < self.ray_dist_threshold]
            inter_points_1 = points_1[vals_1 < self.ray_dist_threshold]
            inter_points = torch.cat([inter_points_0, inter_points_1])
            mean_l.append(inter_points.mean(dim=0))
            var_l.append(inter_points.var(dim=0))
        mean_tor = torch.vstack(mean_l)
        var_tor = torch.vstack(var_l)
        return (mean_tor, var_tor)

    def fit_continuous_function(
        self, mean_tor, var_tor, n_iter=50000, n_display=5000, lr_mean=1e-4, lr_std=1e-4
    ):
        self.weights.requires_grad_()
        opt = torch.optim.Adam({self.weights}, lr=lr_mean)
        print("Matching trajectory distribution mean")
        for i in range(n_iter):
            opt.zero_grad()
            xyz = self.feats @ self.weights
            loss = (torch.norm(xyz - mean_tor, dim=-1) ** 2).mean()
            loss.backward()
            opt.step()
            if i % n_display == 0:
                print("{}:{}".format(i, loss))

        self.std_weights.requires_grad_()
        opt = torch.optim.Adam({self.std_weights}, lr=lr_std)
        print("Matching trajectory distribution std")
        for i in range(n_iter):
            opt.zero_grad()
            xyz_vars = self.feats @ nn.Softplus()(self.std_weights)
            loss = (torch.norm(xyz_vars - torch.sqrt(var_tor), dim=-1) ** 2).mean()
            loss.backward()
            opt.step()
            if i % n_display == 0:
                print("{}:{}".format(i, loss))

    def generate_trajectories(self, n_traj_gen=50, number_t_steps=200):
        # we can have more dense trajectory timesteps here
        t_check_test = torch.linspace(0, 1, number_t_steps).reshape((-1, 1))
        t_check_test = t_check_test.to(self.device)
        feats_test = rbf_kernel(t_check_test, self.t_s, self.gamma)
        xyz_list = []
        for ii in range(n_traj_gen):
            weights_tr = self.weights + nn.Softplus()(
                self.std_weights
            ) * torch.randn_like(self.std_weights)
            xyz = feats_test @ weights_tr
            xyz_list.append(xyz.detach().clone()[None])
        trajs_generated = torch.vstack(xyz_list)
        return trajs_generated

    def sample_weights(self, n_traj_gen=50):
        std_weights_tiled = torch.tile(self.std_weights[None], (n_traj_gen, 1, 1))
        weights_drawn = torch.tile(
            self.weights[None], (n_traj_gen, 1, 1)
        ) + nn.Softplus()(std_weights_tiled) * torch.randn_like(std_weights_tiled)
        return weights_drawn

    def condition_start(self, start_pos, n_traj_gen=50, number_t_steps=200):
        weights_drawn = self.sample_weights(n_traj_gen)
        start_pos_tiled = torch.tile(start_pos, (n_traj_gen, 1))
        t_check_test = torch.linspace(0, 1, number_t_steps).reshape((-1, 1))
        t_check_test = t_check_test.to(self.device)
        feats_test = rbf_kernel(t_check_test, self.t_s, self.gamma)
        feats_test_tiled = torch.tile(feats_test[None], (n_traj_gen, 1, 1))
        trajs_no_cond = torch.bmm(feats_test_tiled[:, :, 1:], weights_drawn[:, 1:, :])
        # condition now, by adding the first row weights accordingly
        start_diff = start_pos_tiled - trajs_no_cond[:, 0, :]
        weight_diff = start_diff / feats_test_tiled[:, 0, 0][:, None]
        weights_drawn[:, 0, :] = weight_diff
        trajs_cond = torch.bmm(feats_test_tiled, weights_drawn)
        return trajs_cond

    def condition_start_end(
        self, start_pos, end_pos, n_traj_gen=50, number_t_steps=200
    ):
        weights_drawn = self.sample_weights(n_traj_gen)
        start_pos_tiled = torch.tile(start_pos, (n_traj_gen, 1))
        end_pos_tiled = torch.tile(end_pos, (n_traj_gen, 1))
        t_check_test = torch.linspace(0, 1, number_t_steps).reshape((-1, 1))
        t_check_test = t_check_test.to(self.device)
        feats_test = rbf_kernel(t_check_test, self.t_s, self.gamma)
        feats_test_tiled = torch.tile(feats_test[None], (n_traj_gen, 1, 1))
        trajs_no_cond = torch.bmm(
            feats_test_tiled[:, :, 1:-1], weights_drawn[:, 1:-1, :]
        )
        # condition now, by adding the first row weights accordingly
        start_diff = start_pos_tiled - trajs_no_cond[:, 0, :]
        end_diff = end_pos_tiled - trajs_no_cond[:, -1, :]
        start_weight_diff = start_diff / feats_test_tiled[:, 0, 0][:, None]
        end_weight_diff = end_diff / feats_test_tiled[:, -1, -1][:, None]
        weights_drawn[:, 0, :] = start_weight_diff
        weights_drawn[:, -1, :] = end_weight_diff
        trajs_cond = torch.bmm(feats_test_tiled, weights_drawn)
        return trajs_cond
