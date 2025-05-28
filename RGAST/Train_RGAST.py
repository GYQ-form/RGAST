import numpy as np
import os
import scanpy as sc
import anndata
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
import random
from tqdm import tqdm
import warnings
from .RGAST import RGAST
from .utils import Transfer_pytorch_Data, res_search_fixed_clus, Batch_Data, Cal_Spatial_Net, Cal_Expression_Net, mclust_R

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
    
def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

class Train_RGAST:

    def __init__(self, adata, dim_reduction = None, batch_data = False, num_batch_x_y = None, device_idx = 7, spatial_net_arg = {}, exp_net_arg = {}, verbose=True, center_msg='out'):

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

        if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
        elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
        else:
            warnings.warn("No dimentional reduction method specified, using all genes' expression as input.")

        self.dim_reduction = dim_reduction
        self.batch_data = batch_data
        self.adata = adata

        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        if 'Exp_Net' not in adata.uns.keys():
            raise ValueError("Exp_Net is not existed! Run Cal_Expression_Net first!")
        self.data = Transfer_pytorch_Data(adata, dim_reduction=dim_reduction,center_msg=center_msg)
        if verbose:
            print('Size of Input: ', self.data.x.shape)

        if batch_data:
            self.num_batch_x, self.num_batch_y = num_batch_x_y
            Batch_list = Batch_Data(adata, num_batch_x=self.num_batch_x, num_batch_y=self.num_batch_y)
            for temp_adata in Batch_list:
                Cal_Spatial_Net(temp_adata, **spatial_net_arg)
                Cal_Expression_Net(temp_adata, dim_reduce=dim_reduction, **exp_net_arg)
            data_list = [Transfer_pytorch_Data(adata, dim_reduction=dim_reduction,center_msg=center_msg) for adata in Batch_list]
            self.loader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        self.model = None


    def train_RGAST(self, early_stopping = True, label_key = None, save_path = '.', n_clusters = 7, cluster_method = 'leiden',
                    hidden_dims=[100, 32], n_epochs=1000, lr=0.001, key_added='RGAST', att_drop = 0.3,
                    gradient_clipping=5., weight_decay=0.0001, min_epochs=300, random_seed=0, save_loss=False,
                    save_reconstrction=False, save_attention=True):

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
            The dimension of the encoder (depends on RGAST or RGAST2).
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
            If True, the reconstructed PCA profiles are saved in adata.layers['RGAST_ReX'].

        Returns
        -------
        AnnData
        """
        self.save_path = save_path
        self.label_key = label_key
        self.n_clusters = n_clusters

        # seed_everything()
        seed=random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


        if self.model is None:
            model = RGAST(hidden_dims = [self.data.x.shape[1]] + hidden_dims, dim_reduce=self.dim_reduction, att_drop=att_drop).to(self.device)
        else:
            model = self.model.to(self.device)
            
        data = self.data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_list = []
        score_list = [0]
        num_fail = 0

        with tqdm(range(n_epochs)) as tq:
            for epoch in tq:
                if early_stopping:
                    with torch.no_grad():
                        if label_key is not None:
                            if (epoch+1) % 50 == 0:
                                if self.batch_data:
                                    model.to('cpu')
                                    model.eval()
                                    z, _, _, _ = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                                    model.to(self.device)
                                else:
                                    model.eval()
                                    z, _, _, _ = model(data.x, data.edge_index, data.edge_type)
                                z = z.to('cpu').detach().numpy()
                                adata_RGAST = anndata.AnnData(z)
                                adata_RGAST.obs_names=self.adata.obs_names
                                if cluster_method == 'mclust':
                                    mclust_R(adata_RGAST, n_clusters)
                                elif cluster_method == 'leiden':
                                    sc.pp.neighbors(adata_RGAST)
                                    _ = res_search_fixed_clus(adata_RGAST, n_clusters)
                                obs_df = adata_RGAST.obs.join(self.adata.obs[label_key]).dropna(subset=label_key)
                                ARI = adjusted_rand_score(obs_df['RGAST'], obs_df[label_key])
                                if ARI <= max(score_list):
                                    num_fail += 1
                                    if num_fail>3 and epoch>=min_epochs:
                                        break
                                else:
                                    num_fail = 0
                                    torch.save(model,f'{save_path}/model.pth')
                                    self.adata.obs['RGAST'] = adata_RGAST.obs['RGAST']
                                score_list.append(ARI)
                                tq.set_postfix(ARI=round(max(score_list),3))
                        
                        else:
                            if (epoch+1) % 50 == 0:
                                if self.batch_data:
                                    model.to('cpu')
                                    model.eval()
                                    z, _, _, _ = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                                    model.to(self.device)
                                else:
                                    model.eval()
                                    z, _, _, _ = model(data.x, data.edge_index, data.edge_type)
                                z = z.to('cpu').detach().numpy()
                                adata_RGAST = anndata.AnnData(z)
                                adata_RGAST.obs_names=self.adata.obs_names
                                sc.pp.neighbors(adata_RGAST)
                                _ = res_search_fixed_clus(adata_RGAST, n_clusters)
                                SC = silhouette_score(z, adata_RGAST.obs['RGAST'])
                                if SC <= max(score_list):
                                    num_fail += 1
                                    if num_fail>3 and epoch>=min_epochs:
                                        break
                                else:
                                    num_fail = 0
                                    torch.save(model,f'{save_path}/model.pth')
                                    self.adata.obs['RGAST'] = adata_RGAST.obs['RGAST']
                                score_list.append(SC)
                                tq.set_postfix(SC=round(max(score_list),3))
                
                if self.batch_data:
                    for batch in self.loader:
                        batch = batch.to(self.device)
                        model.train()
                        optimizer.zero_grad()
                        z, out, _, _ = model(batch.x, batch.edge_index, batch.edge_type)
                        loss = F.mse_loss(batch.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        loss_list.append(loss.item())
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()

                else:
                    model.train()
                    optimizer.zero_grad()
                    z, out, _, _ = model(data.x, data.edge_index, data.edge_type)
                    loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    loss_list.append(loss.item())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()
        
        if early_stopping == True and os.path.exists(f'{save_path}/model.pth'):
            model = torch.load(f'{save_path}/model.pth').to(self.device)

        with torch.no_grad():
            if self.batch_data:
                model.to('cpu')
                model.eval()
                z, out, att1, att2 = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                model.to(self.device)
            else:
                model.eval()
                z, out, att1, att2 = model(data.x, data.edge_index, data.edge_type)

        RGAST_rep = z.to('cpu').detach().numpy()
        np.save(f'{save_path}/RGAST_embedding.npy', RGAST_rep)
        torch.save(model.to('cpu'),f'{save_path}/model.pth')
        self.adata.obsm[key_added] = RGAST_rep

        if save_loss:
            self.adata.uns['RGAST_loss'] = loss_list
        if save_reconstrction:
            ReX = out.to('cpu').numpy()
            self.adata.obsm['RGAST_ReX'] = ReX
        if save_attention:
            self.adata.uns['att1'] = (att1[0].to('cpu').numpy(),att1[1].to('cpu').numpy())
            self.adata.uns['att2'] = (att2[0].to('cpu').numpy(),att2[1].to('cpu').numpy())

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
        y_pred_last = np.array(self.adata.obs['RGAST'],dtype=np.int32).copy()
        counts = len(np.bincount(y_pred_last))
        cluster_layer = []
        for i in range(counts):
            cluster_layer.append(np.mean(test_z[y_pred_last==i,],axis=0))
        cluster_layer = torch.tensor(cluster_layer).to(self.device)
        data = self.data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        score_list = [0]
        num_fail = 0

        with tqdm(range(num_epochs)) as tq:
            for epoch_id in tq:

                if (epoch_id+1) % dec_interval == 0:

                    if early_stopping:
                        #early stopping
                        if self.label_key is not None:
                            model.eval()
                            z, _ = model(data.x, data.edge_index, data.edge_type)
                            z = z.to('cpu').detach().numpy()
                            adata_RGAST = anndata.AnnData(z)
                            adata_RGAST.obs_names=self.adata.obs_names
                            sc.pp.neighbors(adata_RGAST)
                            _ = res_search_fixed_clus(adata_RGAST, self.n_clusters)
                            obs_df = adata_RGAST.obs.join(self.adata.obs[self.label_key]).dropna(subset=self.label_key)
                            ARI = adjusted_rand_score(obs_df['RGAST'], obs_df[self.label_key])
                            if ARI <= max(score_list):
                                num_fail += 1
                                if num_fail>3 and epoch_id>=300:
                                    break
                            else:
                                num_fail = 0
                                torch.save(model,f'{self.save_path}/model.pth')
                                self.adata.obs['RGAST'] = adata_RGAST.obs['RGAST']
                            score_list.append(ARI)
                            tq.set_postfix(ARI=round(max(score_list),3))

                        else:
                            model.eval()
                            z, _ = model(data.x, data.edge_index, data.edge_type)
                            z = z.to('cpu').detach().numpy()
                            adata_RGAST = anndata.AnnData(z)
                            adata_RGAST.obs_names=self.adata.obs_names
                            sc.pp.neighbors(adata_RGAST)
                            _ = res_search_fixed_clus(adata_RGAST, self.n_clusters)
                            SC = silhouette_score(z, adata_RGAST.obs['RGAST'])
                            if SC <= max(score_list):
                                num_fail += 1
                                if num_fail>3 and epoch_id>=300:
                                    break
                            else:
                                num_fail = 0
                                torch.save(model,f'{self.save_path}/model.pth')
                                self.adata.obs['RGAST'] = adata_RGAST.obs['RGAST']
                            score_list.append(SC)
                            tq.set_postfix(SC=round(max(score_list),3))

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
        self.model = torch.load(path, map_location=self.device)

    def save_model(self, path):
        torch.save(self.model,f'{path}/model.pth')

    @torch.no_grad()
    def process(self, gdata = None):
        if gdata is None:
            gdata = self.data
        self.model.to(self.device)
        self.model.eval()
        gdata = gdata.to(self.device)
        return self.model(gdata.x, gdata.edge_index, gdata.edge_type)

