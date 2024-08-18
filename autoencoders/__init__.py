import numpy as np

from .ae import AE, NRAE, DCEC
from .modules import FC_vec

""" FC_image """


def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        linear_layer_type = kwargs["linear_layer_type"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            linear_layer_type=linear_layer_type,
        )
    else:
        raise ValueError(f"Network architecture {kwargs['arch']} not available")

    """ elif kwargs["arch"] == "fc_image":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_image(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        ) """
    return net


def get_ae(**model_cfg_):
    model_cfg = model_cfg_["model_cfg"]
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    if model_cfg["arch"] == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = AE(encoder, decoder, config=model_cfg_)
    elif model_cfg["arch"] == "nrael":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = NRAE(
            encoder,
            decoder,
            approx_order=1,
            kernel=model_cfg["kernel"],
            config=model_cfg_,
        )
    elif model_cfg["arch"] == "nraeq":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = NRAE(
            encoder,
            decoder,
            approx_order=2,
            kernel=model_cfg["kernel"],
            config=model_cfg_,
        )
    elif model_cfg["arch"] == "dcec":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = DCEC(
            encoder=encoder,
            decoder=decoder,
            z_dim=z_dim,
            n_clusters=model_cfg["n_clusters"],
            config=model_cfg_,
        )
    else:
        raise ValueError(f"Autoencoder model {model_cfg['arch']} not available")
    return ae
