import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import scanpy as sc
import torch
from torch_geometric.data import Data
import contextlib
import io
import warnings

def silence_output(func):
    def wrapper(*args, **kwargs):
        # 创建一个空的字符串IO对象
        empty_io = io.StringIO()

        # 使用redirect_stdout和redirect_stderr将输出重定向到空的IO对象
        with contextlib.redirect_stdout(empty_io), contextlib.redirect_stderr(empty_io):
            result = func(*args, **kwargs)

        return result

    return wrapper

def refine_spatial_cluster(adata, pred):
    # 获取空间网络
    G_df = adata.uns['Spatial_Net'].copy()
    # 创建cell名称到索引的映射
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # 将cell名称映射到索引
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # 创建稀疏矩阵
    G = sp.coo_matrix((G_df['Distance'], (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # 转换为CSR格式以提高访问效率
    G = G.tocsr()
    pred = np.array(pred)
    refined_pred = pred.copy()
    for i in range(adata.n_obs):
        # 获取当前cell的邻居及其距离
        neighbors = G[i].nonzero()[1]  # 获取非零元素的列索引
        num_nbs = len(neighbors)
        if num_nbs == 0:
            continue
        distances = G[i, neighbors].toarray().flatten()
        # 排序并选择最近的邻居
        sorted_idx = np.argsort(distances)
        nearest_neighbors = neighbors[sorted_idx][:num_nbs]
        # 获取邻居的预测值
        nbs_pred = pred[nearest_neighbors]
        self_pred = pred[i]
        # 统计邻居中的预测值
        v_c = pd.Series(nbs_pred).value_counts()
        # 决定是否修改当前cell的预测值
        if (v_c.get(self_pred, 0) < num_nbs / 2) and (v_c.max() >= num_nbs / 2):
            refined_pred[i] = v_c.idxmax()
    return refined_pred

def plot_clustering(adata, colors, title = None, savepath = None):
    adata.obs['x_pixel'] = adata.obsm['spatial'][:, 0]
    adata.obs['y_pixel'] = adata.obsm['spatial'][:, 1]

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    sc.pl.scatter(adata, alpha=1, x="x_pixel", y="y_pixel", color=colors, title=title,
                  palette=sns.color_palette('plasma', 7), show=False, ax=ax1)

    ax1.set_aspect('equal', 'box')
    ax1.axis('off')
    ax1.axes.invert_yaxis()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')

def Transfer_pytorch_Data(adata, dim_reduction=None, center_msg='out'):
    """\
    Construct graph data for training.

    Parameters
    ----------
    adata
        AnnData object which contains Spatial network and Expression network.
    dim_reduction
        Dimensional reduction methods (or the input feature). Can be 'PCA', 
        'HVG' or None (default using all gene expression, which may cause out of memeory when training).
    center_msg
        Message passing mode through the graph. Given a center spot, 
        'in' denotes that the message is flowing from connected spots to the center spot,
        'out' denotes that the message is flowing from the center spot to the connected spots.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    #Expression edge
    G_df = adata.uns['Exp_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    exp_edge = np.nonzero(G)

    #Spatial edge
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    spatial_edge = np.nonzero(G)
    
    if dim_reduction=='PCA':
        feat = adata.obsm['X_pca']
    elif dim_reduction=='HVG':
        adata_Vars = adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()
        else:
            feat = adata_Vars.X
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()
        else:
            feat = adata.X

    if center_msg=='out':
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[0],spatial_edge[0])),
            np.concatenate((exp_edge[1],spatial_edge[1]))])).contiguous(), 
            x=torch.FloatTensor(feat.copy()))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[1],spatial_edge[1])),
            np.concatenate((exp_edge[0],spatial_edge[0]))])).contiguous(), 
            x=torch.FloatTensor(feat.copy()))
    edge_type = torch.zeros(exp_edge[0].shape[0]+spatial_edge[0].shape[0],dtype=torch.int64)
    edge_type[exp_edge[0].shape[0]:] = 1
    data.edge_type = edge_type
        
    return data

def Batch_Data(adata, num_batch_x, num_batch_y, plot_Stats=False):
    # 提取所需的空间坐标数据并转换为 numpy 数组
    Sp_df = adata.obsm['spatial']

    # 计算分批的坐标范围
    batch_x_coor = np.percentile(Sp_df[:, 0], np.linspace(0, 100, num_batch_x + 1))
    batch_y_coor = np.percentile(Sp_df[:, 1], np.linspace(0, 100, num_batch_y + 1))

    Batch_list = []
    for it_x in range(num_batch_x):
        min_x, max_x = batch_x_coor[it_x], batch_x_coor[it_x + 1]
        for it_y in range(num_batch_y):
            min_y, max_y = batch_y_coor[it_y], batch_y_coor[it_y + 1]

            # 使用布尔索引进行空间坐标过滤
            mask_x = (Sp_df[:, 0] >= min_x) & (Sp_df[:, 0] <= max_x)
            mask_y = (Sp_df[:, 1] >= min_y) & (Sp_df[:, 1] <= max_y)
            mask = mask_x & mask_y

            # 生成子集并添加到列表中
            temp_adata = adata[mask].copy()
            if temp_adata.shape[0] > 10:
                Batch_list.append(temp_adata)
            
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=6, model='KNN', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('Spatial graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Cal_Expression_Net(adata, k_cutoff=6, dim_reduce=None, verbose=True):

    if verbose:
        print('------Calculating Expression simalarity graph...')

    if dim_reduce=='PCA':
        coor = pd.DataFrame(adata.obsm['X_pca'])
        coor.index = adata.obs.index
    elif dim_reduce=='HVG':
        adata_Vars = adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()
        else:
            feat = adata_Vars.X
        coor = pd.DataFrame(feat)
        coor.index = adata.obs.index
        coor.columns = adata.var_names[adata.var['highly_variable']]
        adata.obsm['HVG'] = coor
    else:
        warnings.warn("No dimentional reduction method specified, using all genes' expression to calculate expression similarity network.")
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()
        else:
            feat = adata.X
        coor = pd.DataFrame(feat)
        coor.index = adata.obs.index
        coor.columns = adata.var_names

    n_nbrs = k_cutoff+1 if k_cutoff+1<coor.shape[0] else coor.shape[0]
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_nbrs).fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    exp_Net = KNN_df.copy()
    exp_Net = exp_Net.loc[exp_Net['Distance']>0,]

    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    exp_Net['Cell1'] = exp_Net['Cell1'].map(id_cell_trans)
    exp_Net['Cell2'] = exp_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('Expression graph contains %d edges, %d cells.' %(exp_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(exp_Net.shape[0]/adata.n_obs))

    adata.uns['Exp_Net'] = exp_Net
    

def cal_metagene(adata,gene_list,obs_name='metagene',layer=None):

    # 提取感兴趣基因的表达矩阵
    if layer is not None:
        gene_expressions = adata[:, gene_list].layers[layer]
    else:
        gene_expressions = adata[:, gene_list].X

    # 如果是稀疏矩阵，则转换为密集矩阵
    if sp.issparse(gene_expressions):
        gene_expressions = gene_expressions.toarray()

    # 计算给定基因列表的表达值之和
    metagene_expression = np.sum(gene_expressions, axis=1)

    # 将新的 metagene 添加到 anndata 对象中
    adata.obs[obs_name] = metagene_expression


def res_search_fixed_clus(adata, fixed_clus_count, max_res=2.5, min_res=0, increment=0.02, key_added='RGAST'):
    for res in np.arange(max_res, min_res, -increment):
        sc.tl.leiden(adata, random_state=2024, resolution=res,key_added=key_added)
        count_unique_leiden = len(adata.obs[key_added].unique())
        if count_unique_leiden <= fixed_clus_count:
            break
    return res


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.
    
    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section, ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat(
            [adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section-1):
        section_1 = section_order[it]
        section_2 = section_order[it+1]
        if verbose:
            print('------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        Z_Net_ID = section_1+'-'+section_2
        temp_adata = adata[adata.obs[key_section].isin(
            [section_1, section_2]), ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge, ]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, [
            'Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat(
        [adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
            (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
            (adata.uns['Spatial_Net'].shape[0]/adata.n_obs))


def Cal_Expression_3D(adata, k_cutoff=6, key_section='Section_id', verbose=True):

    adata.uns['Exp_Net'] = pd.DataFrame()
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating Expression Network of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section, ].copy()
        sc.pp.filter_genes(temp_adata, min_cells=5)
        sc.pp.normalize_total(temp_adata, target_sum=1, exclude_highly_expressed=True)
        sc.pp.scale(temp_adata)
        sc.pp.pca(temp_adata, n_comps=100)
        Cal_Expression_Net(
            temp_adata, k_cutoff=k_cutoff)
        temp_adata.uns['Exp_Net']['SNN'] = temp_section
        adata.uns['Exp_Net'] = pd.concat(
            [adata.uns['Exp_Net'], temp_adata.uns['Exp_Net']])


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)


@silence_output
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm=None, random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    if used_obsm is None:
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(np.array(adata.X)), num_cluster, modelNames)
    else:
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
       