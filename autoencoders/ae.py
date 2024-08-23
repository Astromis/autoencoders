import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from warnings import warn
from pathlib import Path
import yaml
from itertools import combinations
import faiss


from .layers import ClusteringLayer
from .dataset import EmbeddingDataset
from .dataset import EmbeddingDatasetWithGraph
from .utils import get_kernel_function
from .losses import RankNetLoss

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, config=None):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss = torch.nn.MSELoss()
        self.config = config

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def validation_step(self, x, **kwargs):
        recon = self.forward(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

    def train_step(self, batch, optimizer, **kwargs):
        x, y = batch
        x = x.to("cuda")
        optimizer.zero_grad()
        recon = self.forward(x)
        loss = self.mse_loss(recon, x)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def apply_encoder(self, data, batch_size=1000):
        data = self.prepare_dataloader(data, batch_size)
        vectors = []
        self.encoder = self.encoder.eval()
        with torch.no_grad():
            with tqdm(data, unit="batch") as tbatch:
                for batch in tbatch:
                    x, y = batch
                    x = x.to("cuda")
                    tbatch.set_description("Applying")
                    vectors.append(self.encoder(x).detach().cpu().numpy())
        self.encoder = self.encoder.train()
        return np.vstack(vectors)

    def prepare_dataloader(self, data, batch_size):
        if not isinstance(data, EmbeddingDataset):
            data = EmbeddingDataset(data)
        data = DataLoader(data, batch_size=batch_size)
        return data

    def train(self, data, epoch, batch_size=100, lr=1e-3):
        data = self.prepare_dataloader(data, batch_size)
        params_to_optimize = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
        for _ in range(epoch):
            with tqdm(data, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {_}")
                    metrics = self.train_step(batch, optim)
                    tepoch.set_postfix(loss=metrics["loss"])

    def save_model(self, save_path, rewrite=True):
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(exist_ok=rewrite, parents=True)
        torch.save(self.state_dict(), save_path / "model_weights.bin")
        if self.config is not None:
            with open(save_path / "config.yml", "w") as f:
                yaml.safe_dump(self.config, f)
        else:
            warn("Your model doesn't have a config.")

    @classmethod
    def load_model(cls, load_path):
        from . import get_ae

        if isinstance(load_path, str):
            load_path = Path(load_path)
        with open(load_path / "config.yml") as f:
            model_config = yaml.safe_load(f)
        model = get_ae(**model_config)
        model.load_state_dict(torch.load(load_path / "model_weights.bin"))
        return model


class NRAE(Autoencoder):
    def __init__(self, encoder, decoder, kernel_config, graph_config, approx_order=1, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.config = graph_config
        self.approx_order = approx_order
        self.kernel_func = get_kernel_function(kernel_config)

    def jacobian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
        inputs = (
            z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim)
        )  # (bs * num_nn , z_dim)
        jac = torch.autograd.functional.jvp(
            self.decoder, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, num_nn, -1)
        return jac

    def prepare_dataloader(self, data, batch_size):
        if not isinstance(data, EmbeddingDataset):
            data = EmbeddingDatasetWithGraph(data, self.config)
        data = DataLoader(data, batch_size=batch_size)
        return data

    def jacobian_and_hessian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
        inputs = (
            z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim)
        )  # (bs * num_nn , z_dim)

        def jac_temp(inputs):
            jac = torch.autograd.functional.jvp(
                self.decoder, inputs, v=v, create_graph=create_graph
            )[1].view(batch_size, num_nn, -1)
            return jac

        temp = torch.autograd.functional.jvp(
            jac_temp, inputs, v=v, create_graph=create_graph
        )

        jac = temp[0].view(batch_size, num_nn, -1)
        hessian = temp[1].view(batch_size, num_nn, -1)
        return jac, hessian

    def neighborhood_recon(self, z_c, z_nn):
        recon = self.decoder(z_c)
        recon_x = recon.view(z_c.size(0), -1).unsqueeze(1)  # (bs, 1, x_dim)
        dz = z_nn - z_c.unsqueeze(1)  # (bs, num_nn, z_dim)
        if self.approx_order == 1:
            Jdz = self.jacobian(z_c, dz)  # (bs, num_nn, x_dim)
            n_recon = recon_x + Jdz
        elif self.approx_order == 2:
            Jdz, dzHdz = self.jacobian_and_hessian(z_c, dz)
            n_recon = recon_x + Jdz + 0.5 * dzHdz
        return n_recon

    def train_step(self, batch, optimizer, **kwargs):
        x_c, x_nn = batch
        optimizer.zero_grad()
        bs = x_nn.size(0)
        num_nn = x_nn.size(1)

        z_c = self.encoder(x_c)
        z_dim = z_c.size(1)
        z_nn = self.encoder(x_nn.view([-1] + list(x_nn.size()[2:]))).view(bs, -1, z_dim)
        n_recon = self.neighborhood_recon(z_c, z_nn)
        n_loss = torch.norm(x_nn.view(bs, num_nn, -1) - n_recon, dim=2) ** 2
        weights = self.kernel_func(x_c, x_nn)
        loss = (weights * n_loss).mean()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}


class DCEC(Autoencoder):
    def __init__(
        self, encoder, decoder, z_dim, n_clusters, update_interval=4, tol=1e-3, **kwargs
    ):
        super().__init__(encoder, decoder, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.update_interval = update_interval
        self.n_clusters = n_clusters
        self.clustering_layer = ClusteringLayer(z_dim, self.n_clusters,)
        self.clustering_layer = self.clustering_layer.to("cuda")
        self.tol = tol
        self.loss_kld = torch.nn.KLDivLoss()
        self.loss_mse = torch.nn.MSELoss()
        self.p = None

    def init_clustering_layer(self, data):
        encoded = self.apply_encoder(data)
        kmeans = faiss.Kmeans(encoded.shape[1], self.n_clusters, niter=25)
        kmeans.train(encoded)
        y_pred = kmeans.index.search(encoded, 1)[1].squeeze()
        self.clustering_layer.init_weights(
            torch.from_numpy(kmeans.centroids).to("cuda")
        )  # should be (self.n_clusters, input_dim)
        return y_pred

    def target_distribution(self, q):
        weight = q ** 2 / torch.sum(q, dim=0)
        return torch.transpose(
            torch.transpose(weight, 0, 1) / torch.sum(weight, dim=1), 0, 1
        )

    def upldate_aux_distribution(self, dataloader, ite):
        with torch.no_grad():
            q_stack = []
            for x, y in dataloader:
                x = x.to("cuda")
                _, q = self.forward(x, include_clusters=True)
                q_stack.append(q)
            q = torch.vstack(q_stack)
            self.p = self.target_distribution(q)
        y_pred = q.argmax(1)
        return y_pred

    def train_step_dcec(self, batch, optimizer, iter_num):
        x, y = batch
        batch_size = x.shape[0]
        x = x.to("cuda")
        optimizer.zero_grad()
        out, out_clust = self.forward(x, include_clusters=True)
        l1 = self.loss_kld(
            out_clust,
            self.p[batch_size * iter_num : batch_size * iter_num + batch_size],
        )
        l2 = self.loss_mse(out, x)
        l = l1 + l2
        l.backward()
        optimizer.step()
        loss = l.data.detach().cpu().numpy()
        return {"loss": loss.item()}

    def train(self, data, epoch, batch_size, lr=0.001):
        super(DCEC, self).train(data, 1, batch_size)
        y_pred_last = self.init_clustering_layer(data)
        data = self.prepare_dataloader(data, batch_size)
        params_to_optimize = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
            {"params": self.clustering_layer.parameters()},
        ]
        optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
        for _ in range(epoch):
            with tqdm(data, unit="batch") as tepoch:
                for i, batch in enumerate(tepoch):
                    if i % self.update_interval == 0:
                        y_pred = self.upldate_aux_distribution(data, i)
                        delta_label = (
                            np.sum(y_pred != y_pred_last).astype(np.float32)
                            / y_pred.shape[0]
                        )
                        y_pred_last = np.copy(y_pred.detach().cpu().numpy())
                        # check stop criterion
                        if i > 0 and delta_label < self.tol:
                            print("delta_label ", delta_label, "< tol ", self.tol)
                            print("Reached tolerance threshold. Stopping training.")
                            break
                    tepoch.set_description(f"Epoch {_}")

                    metrics = self.train_step_dcec(batch, optim, i)
                    tepoch.set_postfix(loss=metrics["loss"])
        return
    
    def apply_encoder(self, data, batch_size=1000, include_clusters=False):
        data = self.prepare_dataloader(data, batch_size)
        vectors = []
        clusters = []
        self.encoder = self.encoder.eval()
        self.clustering_layer = self.clustering_layer.eval()
        with torch.no_grad():
            with tqdm(data, unit="batch") as tbatch:
                for batch in tbatch:
                    x, y = batch
                    x = x.to("cuda")
                    tbatch.set_description("Applying")
                    encoded = self.encoder(x)
                    vectors.append(encoded.detach().cpu().numpy())
                    clusters.append(self.clustering_layer(encoded).detach().cpu().numpy())
        self.encoder = self.encoder.train()
        self.clustering_layer  = self.clustering_layer.train()
        if include_clusters:
            return np.vstack(vectors), np.vstack(clusters) 

        else:
            return np.vstack(vectors)
    
    def forward(self, x, include_clusters=False):
        x = self.encoder(x)
        x_clust = self.clustering_layer(x)
        x = self.decoder(x)
        if include_clusters:
            return x, x_clust
        else:
            return x

class DistanceAE(Autoencoder):
    def __init__(self, encoder, decoder, p=2, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.p = p

    def distant_loss(self, orig_batch, encoded_batch):
        orig_d = torch.cdist(orig_batch, orig_batch)  # p=self.p
        encod_d = torch.cdist(encoded_batch, encoded_batch)  # p=self.p
        return torch.sum(torch.pow(orig_d - encod_d, 2))  # self.p

    def train_step(self, batch, **kwargs):
        x, _ = batch
        optim.zero_grad()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        loss_mse = self.mse_loss(decoded, x)
        loss_distance = 1e-3*self.distant_loss(x, encoded)
        loss = loss_distance + loss_mse
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    
class MarginDistanceAE(Autoencoder):
    def __init__(self, encoder, decoder, p=2, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.p = p

    def margin_distant_loss(self, orig_batch, encoded_batch):
        batch_size = orig_batch.shape[0]
        final_sum = []
        for i,j in combinations(range(batch_size), 2):
            final_sum.append(torch.pow((orig_batch[i] - orig_batch[j]) - (encoded_batch[i] - encoded_batch[j]), 2))
        return torch.sum(torch.cat(final_sum))

    def train_step(self, batch, optim, **kwargs):
        x, _ = batch
        x = x.to("cuda")
        optim.zero_grad()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        loss_mse = self.mse_loss(decoded, x)
        loss_distance = 1e-3*self.margin_distant_loss(x, decoded)
        loss = loss_distance + loss_mse
        loss.backward()
        optim.step()
        return {"loss": loss.item()}
    
class RankNetDCEC(DCEC):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.ranknet_loss = RankNetLoss()
        self.scorer = nn.Linear(encoder.net[0].in_features, 1)
        
    def train_step(self, batch, optim, iter_num, **kwargs):
        x, _ = batch
        batch_size = x.shape[0]
        x = x.to("cuda")
        optim.zero_grad()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        encoded_clust = self.clustering_layer(encoded)
        l1 = self.loss_kld(
            encoded_clust,
            self.p[batch_size * iter_num : batch_size * iter_num + batch_size],
        )
        # l2 = self.loss_mse(decoded, x)
        dist = torch.cdist(x, x)
        I = torch.argsort(dist)
        l3 = 0
        for n in range(batch_size):
            index_row = I[n]
            i, j = np.random.randint(batch_size, size=2)
            while i == n or j == n:
                i, j = np.random.randint(batch_size, size=2)
            t_i, t_j = index_row[i], index_row[j]
            s_i = self.scorer(decoded[i])
            s_j = self.scorer(decoded[j])
            l3 += self.ranknet_loss(s_i, s_j, t_i, t_j)
        l3 /= batch_size
        loss = l1 + l3
        loss.backward()
        optim.step()
        return {"loss": loss.item()}
