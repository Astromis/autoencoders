import os
import numpy as np
import torch
import torch.nn as nn
from .layers import ClusteringLayer
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

def get_kernel_function(kernel):
    if kernel['type'] == 'binary':
        def kernel_func(x_c, x_nn):
            '''
            x_c.size() = (bs, dim), 
            x_nn.size() = (bs, num_nn, dim)
            '''
            bs = x_nn.size(0)
            num_nn = x_nn.size(1)
            x_c = x_c.view(bs, -1)
            x_nn = x_nn.view(bs, num_nn, -1)
            eps = 1.0e-12
            index = torch.norm(x_c.unsqueeze(1)-x_nn, dim=2) > eps
            output = torch.ones(bs, num_nn).to(x_c)
            output[~index] = kernel['lambda']
            return output # (bs, num_nn)
    return kernel_func
    
class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = torch.nn.MSELoss()

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
        optimizer.zero_grad()
        recon = self.forward(x)
        loss = self.loss(recon, x)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def apply_encoder(self, data):
        vectors = []
        self.encoder = self.encoder.eval()
        with torch.no_grad():
            with tqdm(data, unit="batch") as tbatch:
                for batch in tbatch:
                    x, y = batch
                    tbatch.set_description("Applying")
                    vectors.append(self.encoder(x).detach().cpu().numpy())
        self.encoder = self.encoder.train()
        return np.vstack(vectors)

    
    def train(self, data, epoch, lr=1e-3):
        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
        optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
        for _ in range(epoch):
            with tqdm(data, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {_}")
                    metrics = self.train_step(batch, optim)
                    tepoch.set_postfix(loss=metrics["loss"])


class NRAE(AE):
    def __init__(self, encoder, decoder, approx_order=1, kernel=None):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.approx_order = approx_order
        self.kernel_func = get_kernel_function(kernel)
    
    def jacobian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
        inputs = (z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim))  # (bs * num_nn , z_dim)
        jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1].view(batch_size, num_nn, -1)
        return jac        

    def jacobian_and_hessian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
        inputs = (z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim))  # (bs * num_nn , z_dim)

        def jac_temp(inputs):
            jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1].view(batch_size, num_nn, -1)
            return jac

        temp = torch.autograd.functional.jvp(jac_temp, inputs, v=v, create_graph=create_graph)

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
            n_recon = recon_x + Jdz + 0.5*dzHdz
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
        n_loss = torch.norm(x_nn.view(bs, num_nn, -1) - n_recon, dim=2)**2
        weights = self.kernel_func(x_c, x_nn)
        loss = (weights*n_loss).mean()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}
        

class DCEC(AE):
    def __init__(self, encoder, decoder, z_dim, n_clusters, update_interval = 4, tol=1e-3):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.update_interval = update_interval
        self.n_clusters = n_clusters
        self.clustering_layer = ClusteringLayer(z_dim, self.n_clusters,)
        self.tol = tol
        self.loss_kld = torch.nn.KLDivLoss()
        self.loss_mse = torch.nn.MSELoss()
        self.p = None
    
    def init_clustering_layer(self, data):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.apply_encoder(data))
        self.clustering_layer.init_weights(torch.from_numpy(kmeans.cluster_centers_)) # should be (self.n_clusters, input_dim)
        return y_pred
    
    def target_distribution(self, q):
        weight = q ** 2 / torch.sum(q, dim=0)
        return torch.transpose(torch.transpose(weight, 0, 1) / torch.sum(weight, dim=1), 0,1)
    
    def upldate_aux_distribution(self, dataloader, ite):
        with torch.no_grad():
            q_stack = []
            for x,y in dataloader:
                _, q = self.forward(x)
                q_stack.append(q)
            q = torch.vstack(q_stack)
            self.p = self.target_distribution(q)
        y_pred = q.argmax(1)
        return y_pred 

    def train_step(self, batch, optimizer, batch_size, iter_num):
        x,y = batch
        optimizer.zero_grad()
        out, out_clust = self.forward(x)
        l1 = self.loss_kld(out_clust, self.p[batch_size * iter_num:batch_size * iter_num + batch_size])
        #l1.backward()
        l2 = self.loss_mse(out, x)
        #l2.backward()
        l = 0.1*l1 + l2
        loss = l.data.detach().numpy()
        l.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def train(self, data, epoch, lr=0.001):
        # super().train(data, epoch, lr=lr)
        
        y_pred_last = self.init_clustering_layer(data)
        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()},
            {"params": self.clustering_layer.parameters()}
        ]
        optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
        for _ in range(epoch):
            with tqdm(data, unit="batch") as tepoch:
                for i, batch in enumerate(tepoch):
                    if i % self.update_interval == 0:
                        y_pred = self.upldate_aux_distribution(data, i)
                        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                        y_pred_last = np.copy(y_pred)
                        # check stop criterion
                        if i > 0 and delta_label < self.tol:
                            print('delta_label ', delta_label, '< tol ', self.tol)
                            print('Reached tolerance threshold. Stopping training.')
                            break
                    tepoch.set_description(f"Epoch {_}")

                    metrics = self.train_step(batch, optim, data.batch_size, i)
                    tepoch.set_postfix(loss=metrics["loss"])
        return 

    def forward(self, x):
        x = self.encoder(x)
        x_clust = self.clustering_layer(x)
        x = self.decoder(x)
        return x, x_clust