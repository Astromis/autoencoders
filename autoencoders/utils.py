import torch


def get_kernel_function(kernel_config):
    if kernel_config["type"] == "binary":

        def kernel_func(x_c, x_nn):
            """
            x_c.size() = (bs, dim), 
            x_nn.size() = (bs, num_nn, dim)
            """
            bs = x_nn.size(0)
            num_nn = x_nn.size(1)
            x_c = x_c.view(bs, -1)
            x_nn = x_nn.view(bs, num_nn, -1)
            eps = 1.0e-12
            index = torch.norm(x_c.unsqueeze(1) - x_nn, dim=2) > eps
            output = torch.ones(bs, num_nn).to(x_c)
            output[~index] = kernel_config["lambda"]
            return output  # (bs, num_nn)

    return kernel_func
