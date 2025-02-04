# Wzhi: Simple controller to track eef coordinates.
import pytorch_kinematics as pk
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class controller:
    def __init__(self, robot_urdf_path, eef_name):
        self.chain = chain = pk.build_serial_chain_from_urdf(
            open(robot_urdf_path).read(), eef_name
        )
        self.chain = self.chain.to(device=device)

    def attractor_dist(self, q, x_goal):
        q.requires_grad_()
        eef_pose = self.chain.forward_kinematics(q)
        eef_pos = eef_pose.get_matrix()[0, :3, 3]
        return torch.norm(eef_pos - x_goal)

    def get_vel_q(self, q, x_goal):
        dist = self.attractor_dist(q, x_goal)
        q_vel = torch.autograd.grad(dist, q, create_graph=True)[0]
        return q_vel

    def integrate_to_target(self, q, x_goal, max_t=1000, dt=0.025, eps=0.05):
        cur_q = q.clone()
        q_list = [cur_q]
        for i in range(max_t):
            eef_pose = self.chain.forward_kinematics(cur_q)
            eef_pos = eef_pose.get_matrix()[0, :3, 3]
            if torch.norm(eef_pos - x_goal) < eps:
                break
            q_vel = self.get_vel_q(cur_q, x_goal)
            cur_q = cur_q - q_vel * dt
            cur_q = cur_q.detach()
            q_list.append(cur_q[None].clone())
        return torch.vstack(q_list)

    def follow_traj(self, q_init, xs, max_t=500, dt=0.02, eps=0.03):
        cur_q = q_init.clone()
        q_all_list = []
        for goal_ind in range(len(xs)):
            x_goal = xs[goal_ind]
            q_vels = self.integrate_to_target(
                cur_q, x_goal, max_t=max_t, dt=dt, eps=eps
            )
            cur_q = q_vels[-1]
            q_all_list.append(q_vels[:-1].detach().clone())
        return torch.vstack(q_all_list)
