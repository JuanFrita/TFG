import torch
from torch.nn import Module

class Post_Prob_Val(Module):
    def __init__(self, sigma, stride, background_ratio, use_background, device):
        super(Post_Prob_Val, self).__init__()
        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background
        self.stride = stride


    def forward(self, points, st_sizes, w_sizes, h_sizes):
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            dis_list = []

            for points_image, w_size, h_size in zip(points, w_sizes, h_sizes):
                if(len(points_image) == 0):
                    dis_list.append([])
                    continue
                assert w_size % self.stride == 0 and h_size % self.stride == 0
                x = points_image[:, 0].unsqueeze_(1)
                y = points_image[:, 1].unsqueeze_(1)
                cood_x = (torch.arange(0, w_size, step=self.stride, dtype=torch.float32, device=self.device) + self.stride / 2).unsqueeze_(0)
                cood_y = (torch.arange(0, h_size, step=self.stride, dtype=torch.float32, device=self.device) + self.stride / 2).unsqueeze_(0)
                x_dis = -2 * torch.matmul(x, cood_x) + x * x + cood_x * cood_x
                y_dis = -2 * torch.matmul(y, cood_y) + y * y + cood_y * cood_y
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1))
                dis_list.append(dis)

            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list
