import numpy as np
import os
import scanpy as sc
import anndata
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from .RGAST import RGAST
from .utils import Transfer_pytorch_Data, res_search_fixed_clus, Batch_Data, Cal_Spatial_Net, Cal_Expression_Net

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

class Train_RGAST:

    def __init__(self, adata, batch_data = False, num_batch_x_y = None, spatial_net_arg = {}, exp_net_arg = {}, verbose=True):

        """\
        Initialization of a RGAST trainer.

        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        num_batch_x_y
            A tuple specifying the number of points at which to segment the spatially transcribed image on the x and y axes.
            Each split is then trained as a batch. This is useful for large scale cases.
        spatial_net_arg
            A dict passing key-word arguments to calculating spatial network in each batch data. See `Cal_Spatial_Net`.
        exp_net_arg
            A dict passing key-word arguments to calculating expression network in each batch data. See `Cal_Expression_Net`
        """

        if 'X_pca' not in adata.obsm.keys():
            raise ValueError("PCA has not been done! Run sc.pp.pca first!")
        if verbose:
            print('Size of Input: ', adata.obsm['X_pca'].shape)
        
        self.batch_data = batch_data
        self.adata = adata
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        if 'Exp_Net' not in adata.uns.keys():
            raise ValueError("Exp_Net is not existed! Run Cal_Expression_Net first!")
        self.data = Transfer_pytorch_Data(adata)

        if batch_data:
            self.num_batch_x, self.num_batch_y = num_batch_x_y
            adata.obs['X'] = adata.obsm['spatial'][:,0]
            adata.obs['Y'] = adata.obsm['spatial'][:,1]
            Batch_list = Batch_Data(adata, num_batch_x=self.num_batch_x, num_batch_y=self.num_batch_y, 
                                    spatial_key=['X', 'Y'])
            for temp_adata in Batch_list:
                Cal_Spatial_Net(temp_adata, **spatial_net_arg)
                Cal_Expression_Net(temp_adata, **exp_net_arg)
            data_list = [Transfer_pytorch_Data(adata) for adata in Batch_list]
            self.loader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None


    def train_RGAST(self, early_stopping = True, label_key = None, save_path = '.', n_clusters = 7,
                    hidden_dims=[100, 32], n_epochs=1000, lr=0.001, key_added='RGAST',
                    gradient_clipping=5., weight_decay=0.0001, verbose=True, 
                    random_seed=0, save_loss=False, save_reconstrction=False):

        """\
        Training graph attention auto-encoder.

        Parameters
        ----------
        early_stopping
            Using early stopping strategy or not. Default = True.
        lable_key
            A key specify the specific column in adata.obs to be treated as reference label.
        save_path
            directory to save the trained RGAST model.
        n_clusters
            number of clusters to set when calculating early stopping criterion.
        hidden_dims
            The dimension of the encoder.
        n_epochs
            Number of total epochs in training.
        lr
            Learning rate for AdamOptimizer.
        key_added
            The latent embeddings are saved in adata.obsm[key_added].
        gradient_clipping
            Gradient Clipping.
        weight_decay
            Weight decay for AdamOptimizer.
        save_loss
            If True, the training loss is saved in adata.uns['RGAST_loss'].
        save_reconstrction
            If True, the reconstructed expression profiles are saved in adata.layers['RGAST_ReX'].
        device
            See torch.device.

        Returns
        -------
        AnnData
        """
        self.save_path = save_path
        self.label_key = label_key
        self.n_clusters = n_clusters

        # seed_everything()
        seed=random_seed
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if self.model is None:
            model = RGAST(hidden_dims = [self.data.x.shape[1]] + hidden_dims).to(self.device)
        else:
            model = self.model.to(self.device)
            
        data = self.data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_list = []
        score_list = [0]
        num_fail = 0
        for epoch in tqdm(range(1, n_epochs+1)):

            if early_stopping:

                if label_key is not None:
                    if epoch % 50 == 0:
                        if self.batch_data:
                            model.to('cpu')
                            model.eval()
                            z, _ = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                            model.to(self.device)
                        else:
                            model.eval()
                            z, _ = model(data.x, data.edge_index, data.edge_type)
                        z = z.to('cpu').detach().numpy()
                        adata_RGAST = anndata.AnnData(z)
                        adata_RGAST.obs_names=self.adata.obs_names
                        sc.pp.neighbors(adata_RGAST)
                        sc.tl.umap(adata_RGAST)
                        _ = res_search_fixed_clus(adata_RGAST, n_clusters)
                        obs_df = adata_RGAST.obs.join(self.adata.obs[label_key]).dropna(subset=label_key)
                        ARI = adjusted_rand_score(obs_df['leiden'], obs_df[label_key])
                        if verbose:
                            print(f'epoch:{epoch},ARI:{ARI}')
                        if ARI <= max(score_list):
                            num_fail += 1
                            if num_fail>3 and epoch>=300:
                                break
                        else:
                            num_fail = 0
                            torch.save(model,f'{save_path}/model.pth')
                            self.adata.obs['leiden'] = adata_RGAST.obs['leiden']
                        score_list.append(ARI)
                
                else:
                    if epoch % 50 == 0:
                        if self.batch_data:
                            model.to('cpu')
                            model.eval()
                            z, _ = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                            model.to(self.device)
                        else:
                            model.eval()
                            z, _ = model(data.x, data.edge_index, data.edge_type)
                        z = z.to('cpu').detach().numpy()
                        adata_RGAST = anndata.AnnData(z)
                        adata_RGAST.obs_names=self.adata.obs_names
                        sc.pp.neighbors(adata_RGAST)
                        sc.tl.umap(adata_RGAST)
                        _ = res_search_fixed_clus(adata_RGAST, n_clusters)
                        SC = silhouette_score(z, adata_RGAST.obs['leiden'])
                        if verbose:
                            print(f'epoch:{epoch},SC:{SC}')
                        if SC <= max(score_list):
                            num_fail += 1
                            if num_fail>3 and epoch>=300:
                                break
                        else:
                            num_fail = 0
                            torch.save(model,f'{save_path}/model.pth')
                            self.adata.obs['leiden'] = adata_RGAST.obs['leiden']
                        score_list.append(SC)
            
            if self.batch_data:
                for batch in self.loader:
                    batch = batch.to(self.device)
                    model.train()
                    optimizer.zero_grad()
                    z, out = model(batch.x, batch.edge_index, batch.edge_type)
                    loss = F.mse_loss(batch.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss_list.append(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()

            else:
                model.train()
                optimizer.zero_grad()
                z, out = model(data.x, data.edge_index, data.edge_type)
                loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss_list.append(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
        
        if os.path.exists(f'{save_path}/model.pth'):
            model = torch.load(f'{save_path}/model.pth').to(self.device)

        if self.batch_data:
            model.to('cpu')
            model.eval()
            z, out = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
            model.to(self.device)
        else:
            model.eval()
            z, out = model(data.x, data.edge_index, data.edge_type)

        RGAST_rep = z.to('cpu').detach().numpy()
        np.save(f'{save_path}/RGAST_embedding.npy', RGAST_rep)
        self.adata.obsm[key_added] = RGAST_rep

        if save_loss:
            self.adata.uns['RGAST_loss'] = loss
        if save_reconstrction:
            ReX = out.to('cpu').detach().numpy()
            self.adata.layers['RGAST_ReX'] = ReX

        self.model = model


    def train_with_dec(self, verbose = True, early_stopping = True, key_added='RGAST', num_epochs=1000, dec_interval=50, dec_tol=0.01):

        """\
        Training graph attention auto-encoder with deep embedding clustering.
        Only call this after call Train_RGAST.train_RGAST() and make sure batch_data = False.

        Parameters
        ----------
        early_stopping
            Using early stopping strategy or not. Default = True.
        key_added
            The latent embeddings are saved in adata.obsm[key_added].
        num_epochs
            Number of total epochs in training.
        dec_interval
            Evaluate after how many epochs (for early stopping).
        dec_tol
            DEC tol.

        Returns
        -------
        AnnData with updated .obsm[key_added]
        """
        
        # initialize cluster parameter
        model = self.model.to(self.device)
        model.eval()
        test_z = self.adata.obsm['RGAST']
        y_pred_last = np.array(self.adata.obs['leiden'],dtype=np.int32).copy()
        counts = len(np.bincount(y_pred_last))
        cluster_layer = []
        for i in range(counts):
            cluster_layer.append(np.mean(test_z[y_pred_last==i,],axis=0))
        cluster_layer = torch.tensor(cluster_layer).to(self.device)
        data = self.data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        score_list = [0]
        num_fail = 0
        for epoch_id in tqdm(range(num_epochs)):

            if early_stopping:

                if epoch_id % dec_interval == 0:
                    #early stopping
                    if self.label_key is not None:
                        model.eval()
                        z, _ = model(data.x, data.edge_index, data.edge_type)
                        z = z.to('cpu').detach().numpy()
                        adata_RGAST = anndata.AnnData(z)
                        adata_RGAST.obs_names=self.adata.obs_names
                        sc.pp.neighbors(adata_RGAST)
                        sc.tl.umap(adata_RGAST)
                        _ = res_search_fixed_clus(adata_RGAST, self.n_clusters)
                        obs_df = adata_RGAST.obs.join(self.adata.obs[self.label_key]).dropna(subset=self.label_key)
                        ARI = adjusted_rand_score(obs_df['leiden'], obs_df[self.label_key])
                        if verbose:
                            print(f'epoch:{epoch_id},ARI:{ARI}')
                        if ARI <= max(score_list):
                            num_fail += 1
                            if num_fail>3 and epoch_id>=300:
                                break
                        else:
                            num_fail = 0
                            torch.save(model,f'{self.save_path}/model.pth')
                            self.adata.obs['leiden'] = adata_RGAST.obs['leiden']
                        score_list.append(ARI)

                    else:
                        model.eval()
                        z, _ = model(data.x, data.edge_index, data.edge_type)
                        z = z.to('cpu').detach().numpy()
                        adata_RGAST = anndata.AnnData(z)
                        adata_RGAST.obs_names=self.adata.obs_names
                        sc.pp.neighbors(adata_RGAST)
                        sc.tl.umap(adata_RGAST)
                        _ = res_search_fixed_clus(adata_RGAST, self.n_clusters)
                        SC = silhouette_score(z, adata_RGAST.obs['leiden'])
                        if verbose:
                            print(f'epoch:{epoch_id},SC:{SC}')
                        if SC <= max(score_list):
                            num_fail += 1
                            if num_fail>3 and epoch_id>=300:
                                break
                        else:
                            num_fail = 0
                            torch.save(model,f'{self.save_path}/model.pth')
                            self.adata.obs['leiden'] = adata_RGAST.obs['leiden']
                        score_list.append(SC)

                    #DEC update
                    z, reconst = model(data.x, data.edge_index, data.edge_type)
                    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2))
                    q = (q.t() / torch.sum(q, 1)).t()
                    tmp_p = target_distribution(torch.Tensor(q))
                    y_pred = tmp_p.cpu().detach().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    if epoch_id > 0 and delta_label < dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break
                

            # training model
            model.train()
            optimizer.zero_grad()
            z, reconst = model(data.x, data.edge_index, data.edge_type)
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2) / 1.0)
            q = (q.t() / torch.sum(q, 1)).t()
            loss_rec = F.mse_loss(data.x, reconst)
            # clustering KL loss
            loss_kl = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = loss_kl + loss_rec
            loss.backward()
            optimizer.step()

        model = torch.load(f'{self.save_path}/model.pth').to(self.device)
        model.eval()
        z, _ = model(data.x, data.edge_index, data.edge_type)

        RGAST_rep = z.to('cpu').detach().numpy()
        np.save(f'{self.save_path}/RGAST_embedding.npy', RGAST_rep)
        self.adata.obsm[key_added] = RGAST_rep
        self.model = model

    def load_model(self, path):
        self.model = torch.load(path)

    def save_model(self, path):
        torch.save(self.model,f'{path}/model.pth')

    def process(self, gdata = None):
        if gdata is None:
            gdata = self.data
        self.model.to(self.device)
        self.model.eval()
        gdata = gdata.to(self.device)
        return self.model(gdata.x, gdata.edge_index, gdata.edge_type)

