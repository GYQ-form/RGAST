from typing import Optional
from typing import Literal
import anndata
import pandas as pd
import numpy as np
import plotly
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D 
import matplotlib as mpl
import scipy.sparse as sp
import pkgutil
import io
from tqdm import tqdm
import scipy
from matplotlib.collections import LineCollection


def ligand_receptor_database(
    database = "CellChat",
    species = "mouse",
    heteromeric_delimiter = "_",
    signaling_type = "Secreted Signaling" # or "Cell-Cell contact" or "ECM-Receptor" or None
):
    """
    Extract ligand-receptor pairs from LR database.

    Parameters
    ----------
    database
        The name of the ligand-receptor database. Use 'CellChat' for CellChatDB [Jin2021]_ of 'CellPhoneDB_v4.0' for CellPhoneDB_v4.0 [Efremova2020]_.
    species
        The species of the ligand-receptor pairs. Choose between 'mouse' and 'human'.
    heteromeric_delimiter
        The character to separate the heteromeric units of heteromeric ligands and receptors. 
        For example, if the heteromeric receptor (TGFbR1, TGFbR2) will be represented as 'TGFbR1_TGFbR2' if this parameter is set to '_'.
    signaling_type
        The type of signaling. Choose from 'Secreted Signaling', 'Cell-Cell Contact', and 'ECM-Receptor' for CellChatDB or 'Secreted Signaling' and 'Cell-Cell Contact' for CellPhoneDB_v4.0. 
        If None, all pairs in the database are returned.

    Returns
    -------
    df_ligrec : pandas.DataFrame
        A pandas DataFrame of the LR pairs with the three columns representing the ligand, receptor, and the signaling pathway name, respectively.

    References
    ----------

    .. [Jin2021] Jin, S., Guerrero-Juarez, C. F., Zhang, L., Chang, I., Ramos, R., Kuan, C. H., ... & Nie, Q. (2021). 
        Inference and analysis of cell-cell communication using CellChat. Nature communications, 12(1), 1-20.
    .. [Efremova2020] Efremova, M., Vento-Tormo, M., Teichmann, S. A., & Vento-Tormo, R. (2020). 
        CellPhoneDB: inferring cell–cell communication from combined expression of multi-subunit ligand–receptor complexes. Nature protocols, 15(4), 1484-1506.

    """

    if database == "CellChat":
        data = pkgutil.get_data(__name__, "_data/LRdatabase/CellChat/CellChatDB.ligrec."+species+".csv")
        df_ligrec = pd.read_csv(io.BytesIO(data), index_col=0)
        if not signaling_type is None:
            df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
    elif database == 'CellPhoneDB_v4.0':
        data = pkgutil.get_data(__name__, "_data/LRdatabase/CellPhoneDB_v4.0/CellPhoneDBv4.0."+species+".csv")
        df_ligrec = pd.read_csv(io.BytesIO(data), index_col=0)
        if not signaling_type is None:
            df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
    
    df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling_type']
    return df_ligrec


def attention_to_interact(adata, attention_key='att2', cum_att_threshold=0.5):
    n_spots = adata.shape[0]
    interact_mat = np.zeros((n_spots,n_spots),dtype=np.int8)
    weighted_mat = np.zeros((n_spots,n_spots))
    attention = adata.uns[attention_key]

    for i in np.arange(n_spots):
        idx = np.where(attention[0][1]==i)[0]
        from_idx = attention[0][0,idx]
        att_score = attention[1][np.where(attention[0][1]==i)].reshape(-1)
        
        #dereplication
        unique_labels, indices = np.unique(from_idx, return_inverse=True)
        summed_values = np.zeros_like(unique_labels, dtype=np.float32)
        for j in range(len(att_score)):
            summed_values[indices[j]] += att_score[j]
        from_idx = unique_labels
        att_score = summed_values

        sorted_indices = np.argsort(att_score)[::-1]
        cumulative_sum = np.cumsum(att_score[sorted_indices])
        stop_index = np.where(cumulative_sum > cum_att_threshold)[0][0]
        used_from_idx = from_idx[sorted_indices[:stop_index+1]]
        used_att = att_score[sorted_indices[:stop_index+1]]
        interact_mat[i,used_from_idx] = interact_mat[used_from_idx,i] = 1
        weighted_mat[used_from_idx,i] = used_att

    adata.obsp['interact_mat'] = interact_mat
    adata.obsp['weighted_mat'] = weighted_mat
    adata.obsp['interact_mat_directed'] = (weighted_mat>0).astype(np.int8)


def attention_to_interact2(adata, attention_key='att2', att_score_threshold=0.3):
    """
    根据设定的注意力分数阈值，构建空间spot之间的交互网络。

    这个函数会处理一个图注意力网络中的注意力分数，并保留所有注意力分数
    高于指定阈值的边，从而生成一个二进制的交互矩阵和一个加权的交互矩阵。

    参数:
    ----------
    adata: AnnData
        包含空间转录组数据的AnnData对象。
        `adata.uns[attention_key]` 应包含注意力数据。
        注意力数据的格式应为一个元组 `(edge_index, edge_weight)`，其中
        `edge_index` 是一个 `[2, n_edges]` 的数组，代表边的连接关系 (源 -> 目标)，
        `edge_weight` 是一个 `[n_edges]` 的数组，代表每条边的注意力分数。
        
    attention_key: str, optional (default='att2')
        在 `adata.uns` 中存储注意力分数的键名。
        
    att_score_threshold: float, optional (default=0.1)
        用于筛选边的注意力分数阈值。只有当一个spot到另一个spot的
        （累加后）注意力分数大于此阈值时，这条边才会被保留。

    返回值:
    -------
    None
        函数会直接修改输入的 `adata` 对象，在 `adata.obsp` 中添加两个稀疏矩阵：
        - 'interact_mat': 二进制交互矩阵，1表示两个spot间存在高于阈值的交互。这是一个对称矩阵。
        - 'weighted_mat': 加权交互矩阵，存储了从源spot到目标spot的有向注意力分数。这是一个非对称矩阵。
    """
    n_spots = adata.shape[0]
    
    interact_row, interact_col = [], []
    weighted_row, weighted_col, weighted_data = [], [], []

    attention = adata.uns[attention_key]
    source_nodes, target_nodes = attention[0][0], attention[0][1]
    edge_weights = attention[1]

    for i in np.arange(n_spots):
        idx = np.where(target_nodes == i)[0]
        
        if len(idx) == 0:
            continue
            
        from_idx = source_nodes[idx]
        att_score = edge_weights[idx].flatten()
        
        unique_from_idx, inverse_indices = np.unique(from_idx, return_inverse=True)
        summed_att_score = np.zeros(unique_from_idx.shape[0], dtype=np.float32)
        
        # 此行现在可以安全执行
        np.add.at(summed_att_score, inverse_indices, att_score)
        
        selected_indices = np.where(summed_att_score > att_score_threshold)[0]
        
        if len(selected_indices) == 0:
            continue

        used_from_idx = unique_from_idx[selected_indices]
        used_att = summed_att_score[selected_indices]
        
        weighted_row.extend(used_from_idx)
        weighted_col.extend([i] * len(used_from_idx))
        weighted_data.extend(used_att)
        
        interact_row.extend(used_from_idx)
        interact_col.extend([i] * len(used_from_idx))
        interact_row.extend([i] * len(used_from_idx))
        interact_col.extend(used_from_idx)

    # --- 构建并存储稀疏矩阵 ---
    weighted_mat = sp.coo_matrix((weighted_data, (weighted_row, weighted_col)), shape=(n_spots, n_spots)).tocsr()
    
    # ==================== (鲁棒性) ====================
    # 增加判断，防止在没有符合条件的边时，因解压空集合而报错。
    if interact_row:
        unique_edges = set(zip(interact_row, interact_col))
        final_interact_row, final_interact_col = zip(*unique_edges)
        interact_data = np.ones(len(final_interact_row))
    else:
        final_interact_row, final_interact_col, interact_data = [], [], []
    # =======================================================

    interact_mat = sp.coo_matrix((interact_data, (final_interact_row, final_interact_col)), shape=(n_spots, n_spots)).tocsr()
    
    adata.obsp['interact_mat'] = interact_mat
    adata.obsp['weighted_mat'] = weighted_mat


def analyze_communication_mechanism(adata: anndata.AnnData, lr_df: pd.DataFrame, adj_matrix: np.ndarray):
    """
    基于已知的细胞通讯网络，量化每个L-R对和通路的贡献。

    Args:
        adata: 已标准化的 AnnData 对象。
        lr_df: 配体-受体数据库 DataFrame。
        adj_matrix: n_obs x n_obs 的0-1邻接矩阵，表示细胞通讯网络。
    """
    print("\n--- Starting Communication Mechanism Analysis ---")
    n_obs = adata.n_obs
    
    # --- 1. 数据准备 ---
    if adj_matrix.shape != (n_obs, n_obs):
        raise ValueError(f"Shape of adj_matrix {adj_matrix.shape} does not match adata.n_obs {n_obs}.")

    # 解析L-R数据库
    lr_df_proc = lr_df.copy()
    lr_df_proc['ligand_subunits'] = lr_df_proc['ligand'].str.split('_')
    lr_df_proc['receptor_subunits'] = lr_df_proc['receptor'].str.split('_')
    
    # 过滤L-R数据库
    all_genes_in_adata = set(adata.var_names)
    def check_presence(row):
        genes = row['ligand_subunits'] + row['receptor_subunits']
        return all(g in all_genes_in_adata for g in genes)
    
    lr_df_proc = lr_df_proc[lr_df_proc.apply(check_presence, axis=1)].reset_index(drop=True)
    print(f"Found {len(lr_df_proc)} L-R pairs with all subunits present in the data.")
    
    # 初始化通路聚合器
    pathway_names = lr_df_proc['pathway'].unique()
    pathway_aggregators = {name: np.zeros((n_obs, n_obs)) for name in pathway_names}
    pathway_counters = {name: 0 for name in pathway_names}
        
    # --- 2. 核心计算 ---
    # 确保adata.X是numpy数组以便索引
    X_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    for _, row in tqdm(lr_df_proc.iterrows(), total=len(lr_df_proc), desc="Processing L-R pairs"):
        ligand_subunits = row['ligand_subunits']
        receptor_subunits = row['receptor_subunits']
        pathway = row['pathway']
        
        # 获取亚基在var中的索引
        ligand_indices = [adata.var_names.get_loc(g) for g in ligand_subunits]
        receptor_indices = [adata.var_names.get_loc(g) for g in receptor_subunits]

        # 计算每个细胞的平均表达 (axis=1)
        L_expr_per_cell = X_matrix[:, ligand_indices].mean(axis=1)
        R_expr_per_cell = X_matrix[:, receptor_indices].mean(axis=1)
        
        # 计算方向性分数
        score_i_sends_j = np.outer(L_expr_per_cell, R_expr_per_cell)
        
        # 合并双向分数 (i->j 和 j->i)
        # score_j_sends_i is score_i_sends_j.T
        # total_lr_score = score_i_sends_j + score_i_sends_j.T
        
        # 关键步骤：使用邻接矩阵进行屏蔽
        masked_lr_score = score_i_sends_j * adj_matrix
        
        # 如果分数矩阵全为0，则不保存，跳过
        if np.all(masked_lr_score == 0):
            continue
            
        # 存储L-R对的结果 (使用稀疏矩阵节省内存)
        lr_key = f"LR_{row['ligand']}_{row['receptor']}"
        adata.obsp[lr_key] = scipy.sparse.csr_matrix(masked_lr_score)
        
        # --- 3. 累加到通路聚合器 ---
        pathway_aggregators[pathway] += masked_lr_score
        pathway_counters[pathway] += 1
        
    # 存储最终的通路聚合结果
    print("\nStoring aggregated pathway scores...")
    for pathway, agg_matrix in pathway_aggregators.items():
        if np.all(agg_matrix == 0):
            continue
        pathway_key = f"pathway_{pathway}"
        agg_matrix = agg_matrix / pathway_counters[pathway] if pathway_counters[pathway] > 0 else agg_matrix
        adata.obsp[pathway_key] = scipy.sparse.csr_matrix(agg_matrix)

    print("\n--- Analysis Finished ---")
    print("Scores have been stored in adata.obsp")


def calculate_pathway_scores(adata: anndata.AnnData) -> anndata.AnnData:
    """
    计算每个通路的全局分数并将其存储在 adata.uns 中。
    分数计算方法为：提取通路矩阵中所有非零值的平均值。

    Args:
        adata: anndata 对象，其 .obsp 中包含键名以 'pathway_' 开头的稀疏矩阵。

    Returns:
        更新后的 anndata 对象。
    """
    print("--- Calculating pathway scores ---")
    
    if not hasattr(adata, 'obsp') or not adata.obsp:
        raise ValueError("adata.obsp is empty. Please run 'RGAST.cci.analyze_communication_mechanism' first.")

    pathway_scores = {}
    
    # 遍历 obsp 中的所有条目
    for key, matrix in adata.obsp.items():
        if key.startswith('pathway_'):
            # 提取通路名称
            pathway_name = key.replace('pathway_', '')
            
            if isinstance(matrix, sp.spmatrix) and hasattr(matrix, 'data'):
                # .data 属性直接存储了所有的非零值
                non_zero_values = matrix.data
                
                if len(non_zero_values) > 0:
                    # 计算非零值的平均值
                    score = np.mean(non_zero_values)
                else:
                    score = 0.0 # 如果没有非零值，分数为0
            else:
                score = matrix.mean()
                    
            pathway_scores[pathway_name] = score
    
    if not pathway_scores:
        print("Warning: No pathway matrices found in adata.obsp (keys starting with 'pathway_').")
        return adata
        
    # 将字典转换为 DataFrame
    score_df = pd.DataFrame(list(pathway_scores.items()), columns=['pathway', 'score'])
    
    # 按分数从高到低排序
    score_df = score_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # 将结果存储在 adata.uns 中
    adata.uns['pathway_score'] = score_df
    
    print(f"Successfully calculated scores for {len(score_df)} pathways.")
    print("Scores stored in adata.uns['pathway_score'].")


def plot_pathway_scores(
    adata: anndata.AnnData, 
    top_n: int = None,
    title: str = "Overall Pathway Communication Strength",
    color: str = "#B19470",
    filename: str = None
):
    """
    使用条形图可视化通路分数。

    Args:
        adata: anndata 对象，其 .uns 中包含 'pathway_score' 键。
        top_n (int, optional): 只显示分数最高的 top_n 个通路。默认为 None (显示全部)。
        title (str, optional): 图像的标题。
        palette (str, optional): seaborn调色板名称。
        filename (str, optional): 如果提供，将图像保存到指定的文件名。
    """
    if 'pathway_score' not in adata.uns:
        raise ValueError("Please run 'RGAST.cci.calculate_pathway_scores' first to compute pathway scores.")
    score_df = adata.uns['pathway_score']

    if top_n:
        plot_data = score_df.head(top_n)
    else:
        plot_data = score_df

    plt.figure(figsize=(8, max(6, len(plot_data) * 0.4))) # 图形高度自适应
    
    # 创建条形图，Seaborn会自动使用DataFrame的顺序
    sns.barplot(x='score', y='pathway', data=plot_data, color=color)
    
    plt.tick_params(labelsize=16)  # 设置刻度标签大小
    plt.title(title, fontsize=22, pad=20)
    plt.xlabel("Mean Communication Score", fontsize=20)
    plt.ylabel("Pathway", fontsize=20)
    
    # 移除右边和上边的边框，使图形更美观
    sns.despine()
    
    # 自动调整布局防止标签被遮挡
    plt.tight_layout()

    # Save the plot to a file
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.show()
    plt.close()



def ranked_partial(coord, size):  #size是list，[3,5]代表把总图切成宽3份(x)、高5份(y)的子图
    x_gap = (coord[:,0].max()-coord[:,0].min())/size[0]
    y_gap = (coord[:,1].max()-coord[:,1].min())/size[1]
    x_point = np.arange(coord[:,0].min(), coord[:,0].max(), x_gap).tolist()
    if coord[:,0].max() not in x_point:
        x_point += [coord[:,0].max()]
    y_point = np.arange(coord[:,1].min(), coord[:,1].max(), y_gap).tolist()
    if coord[:,1].max() not in y_point:
        y_point += [coord[:,1].max()]

    x_interval = [[x_point[i],x_point[i+1]] for i in range(len(x_point)) if i!=len(x_point)-1]
    y_interval = [[y_point[i],y_point[i+1]] for i in range(len(y_point)) if i!=len(y_point)-1]

    id_part = []
    subregion_mark = []
    for i in x_interval:
        for j in y_interval:
            id_list = np.where((coord[:,0]>=i[0]) & (coord[:,0]<i[1]) & (coord[:,1]>=j[0]) & (coord[:,1]<j[1]))[0].tolist()  #左开右闭，上开下闭
            id_part.append(id_list)
            subregion_mark.append([i,j])

    return id_part, subregion_mark

def select_celltypes(cell_types, selected_ct, frac):
    indices = np.where(cell_types.isin(selected_ct))[0]
    selected_idx = np.random.choice(indices,round(frac*indices.shape[0]),replace=False)
    return selected_idx

def cal_communication_direction(
    adata: anndata.AnnData,
    network_key: str = 'weighted_mat',
    k: int = 5,
    pos_idx: Optional[np.ndarray] = None,
):
    """
    Construct spatial vector fields for inferred communication.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` x ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    network_key
        The key in ``.obsp`` that contains the weighted directed interaction matrix.
        Default to 'weighted_mat', which is generated by the function:`RGAST.cci.attention_to_interact`.
    k
        Top k senders or receivers to consider when determining the direction.
    pos_idx
        The columns in ``.obsm['spatial']`` to use. If None, all columns are used.
        For example, to use just the first and third columns, set pos_idx to ``numpy.array([0,2],int)``.

    Returns
    -------
    adata : anndata.AnnData
        Vector fields describing signaling directions are added to ``.obsm``, 
    """

    ncell = adata.shape[0]
    pts = np.array(adata.obsm['spatial'], float)
    if not pos_idx is None:
        pts = pts[:,pos_idx]

    S_np = adata.obsp[network_key].toarray() if hasattr(adata.obsp[network_key], 'toarray') else adata.obsp[network_key]
    sender_vf = np.zeros_like(pts)
    receiver_vf = np.zeros_like(pts)

    tmp_idx = np.argsort(-S_np,axis=1)[:,:k]
    avg_v = np.zeros_like(pts)
    for ik in range(k):
        tmp_v = pts[tmp_idx[:,ik]] - pts[np.arange(ncell,dtype=int)]
        tmp_v = normalize(tmp_v, norm='l2')
        avg_v = avg_v + tmp_v * S_np[np.arange(ncell,dtype=int),tmp_idx[:,ik]].reshape(-1,1)
    avg_v = normalize(avg_v)
    sender_vf = avg_v * np.sum(S_np,axis=1).reshape(-1,1)

    S_np = S_np.T
    tmp_idx = np.argsort(-S_np,axis=1)[:,:k]
    avg_v = np.zeros_like(pts)
    for ik in range(k):
        tmp_v = -pts[tmp_idx[:,ik]] + pts[np.arange(ncell,dtype=int)]
        tmp_v = normalize(tmp_v, norm='l2')
        avg_v = avg_v + tmp_v * S_np[np.arange(ncell,dtype=int),tmp_idx[:,ik]].reshape(-1,1)
    avg_v = normalize(avg_v)
    receiver_vf = avg_v * np.sum(S_np,axis=1).reshape(-1,1)

    adata.obsm['sender_vf'] = sender_vf
    adata.obsm['receiver_vf'] = receiver_vf


def plot_cell_communication(
    adata: anndata.AnnData,
    plot_method: str = "cell",
    background: str = "summary",
    background_legend: bool=False,
    clustering: str = None,
    summary: str = "sender",
    cmap: str = "coolwarm",
    cluster_cmap: dict = None,
    pos_idx: np.ndarray = np.array([0,1],int),
    ndsize: float = 1,
    scale: float = 1.0,
    normalize_v: bool = False,
    normalize_v_quantile: float = 0.95,
    arrow_color: str = "#333333",
    grid_density: float = 1.0,
    grid_knn: int = None,
    grid_scale: float = 1.0,
    grid_thresh: float = 1.0,
    grid_width: float = 0.005,
    stream_density: float = 1.0,
    stream_linewidth: float = 1,
    stream_cutoff_perc: float = 5,
    filename: str = None,
    ax: Optional[mpl.axes.Axes] = None
):
    """
    Plot spatial directions of cell-cell communication.

    The cell-cell communication should have been computed by the function :func:`RGAST.cci.attention_to_interact`.
    The cell-cell communication direction should have been computed by the function :func:`RGAST.cci.cal_communication_direction`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` x ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    plot_method
        'cell' plot vectors on individual cells. 
        'grid' plot interpolated vectors on regular grids.
        'stream' streamline plot.
    background
        'summary': scatter plot with color representing total sent or received signal.
        'image': the image in Visium data.
        'cluster': scatter plot with color representing cell clusters.
    background_legend
        Whether to include the background legend when background is set to `summary` or `cluster`.
    clustering
        The key for clustering result. Needed if background is set to `cluster`.
        For example, if ``clustering=='leiden'``, the clustering result should be available in ``.obs['leiden']``.
    summary
        If background is set to 'summary', the numerical value to plot for background.
        'sender': node color represents sender weight.
        'receiver': node color represents receiver weight.
    cmap
        matplotlib colormap name for node summary if numerical (background set to 'summary'), e.g., 'coolwarm'.
        plotly colormap name for node color if summary is (background set to 'cluster'). e.g., 'Alphabet'.
    cluster_cmap
        A dictionary that maps cluster names to colors when setting background to 'cluster'. If given, ``cmap`` will be ignored.
    pos_idx
        The coordinates to use for plotting (2D plot).
    ndsize
        The node size of the spots.
    scale
        The scale parameter passed to the matplotlib quiver function :func:`matplotlib.pyplot.quiver` for vector field plots.
        The smaller the value, the longer the arrows.
    normalize_v
        Whether the normalize the vector field to uniform lengths to highlight the directions without showing magnitudes.
    normalize_v_quantile
        The vector length quantile to use to normalize the vector field.
    arrow_color
        The color of the arrows.
    grid_density
        The density of grid if ``plot_method=='grid'``.
    grid_knn
        If ``plot_method=='grid'``, the number of nearest neighbors to interpolate the signaling directions from spots to grid points.
    grid_scale
        The scale parameter (relative to grid size) for the kernel function of mapping directions of spots to grid points.
    grid_thresh
        The threshold of interpolation weights determining whether to include a grid point. A smaller value gives a tighter cover of the tissue by the grid points.
    grid_width
        The value passed to the ``width`` parameter of the function :func:`matplotlib.pyplot.quiver` when ``plot_method=='grid'``.
    stream_density
        The density of stream lines passed to the ``density`` parameter of the function :func:`matplotlib.pyplot.streamplot` when ``plot_method=='stream'``.
    stream_linewidth
        The width of stream lines passed to the ``linewidth`` parameter of the function :func:`matplotlib.pyplot.streamplot` when ``plot_method=='stream'``.
    stream_cutoff_perc
        The quantile cutoff to ignore the weak vectors. Default to 5 that the vectors shorter than the 5% quantile will not be plotted.
    filename
        If given, save to the filename. For example 'ccc_direction.pdf'.
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.

    """

    if summary == 'sender':
        V = adata.obsm['sender_vf'][:,pos_idx]
        signal_sum = adata.obsp['weighted_mat'].sum(axis=1).reshape(-1,1)
    elif summary == 'receiver':
        V = adata.obsm['receiver_vf'][:,pos_idx]
        signal_sum = adata.obsp['weighted_mat'].sum(axis=0).reshape(-1,1)

    if background=='cluster' and not cmap in ['Plotly','Light24','Dark24','Alphabet']:
        cmap='Alphabet'
    if ax is None:
        fig, ax = plt.subplots()
    if normalize_v:
        V = V / np.quantile(np.linalg.norm(V, axis=1), normalize_v_quantile)
    plot_cell_signaling(
        adata.obsm["spatial"][:,pos_idx],
        V,
        signal_sum,
        cmap = cmap,
        cluster_cmap = cluster_cmap,
        plot_method = plot_method,
        background = background,
        clustering = clustering,
        background_legend = background_legend,
        adata = adata,
        summary = summary,
        scale = scale,
        ndsize = ndsize,
        filename = filename,
        arrow_color = arrow_color,
        grid_density = grid_density,
        grid_knn = grid_knn,
        grid_scale = grid_scale,
        grid_thresh = grid_thresh,
        grid_width = grid_width,
        stream_density = stream_density,
        stream_linewidth = stream_linewidth,
        stream_cutoff_perc = stream_cutoff_perc,
        ax = ax,
        # fig = fig,
    )
    return ax

def plot_cell_signaling(X,
    V,
    signal_sum,
    cmap="coolwarm",
    cluster_cmap = None,
    arrow_color="tab:blue",
    plot_method="cell",
    background='summary',
    clustering=None,
    background_legend=False,
    adata=None,
    summary='sender',
    ndsize = 1,
    scale = 1.0,
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,
    grid_width = 0.005,
    stream_density = 1.0,
    stream_linewidth = 1,
    stream_cutoff_perc = 5,
    filename=None,
    ax = None,
    fig = None
):
    ndcolor = signal_sum
    ncell = X.shape[0]
    
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    if plot_method == "grid" or plot_method == "stream":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)
    
        if grid_knn is None:
            grid_knn = int( X.shape[0] / 50 )
        nn_mdl = NearestNeighbors()
        nn_mdl.fit(X)
        dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        w_sum = w.sum(axis=1)

        V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        V_grid /= np.maximum(1, w_sum)[:,None]

        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            V_grid = V_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid ** 2).sum(0))
            grid_thresh = 10 ** (grid_thresh - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(V_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(V[nbs]),axis=1),axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_perc)

            V_grid[0][cutoff] = np.nan

    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    idx = np.argsort(ndcolor)
    if background == 'summary' or background == 'cluster':
        if not ndsize==0:
            if background == 'summary':
                ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0)
            elif background == 'cluster':
                labels = np.array( adata.obs[clustering], str )
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    if cluster_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cmap[i_label], linewidth=0, label=unique_labels[i_label])
                    elif not cluster_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cluster_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label])
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0,0.0])
        if plot_method == "cell":
            ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], V_cell[:,1], scale=scale, scale_units='x', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], V_grid[:,1], scale=scale, scale_units='x', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid, y_grid, V_grid[0], V_grid[1], color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')
        if plot_method == "cell":
            ax.quiver(X_vec[:,0]*sf, X_vec[:,1]*sf, V_cell[:,0]*sf, V_cell[:,1]*sf, scale=scale, scale_units='x', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0]*sf, grid_pts[:,1]*sf, V_grid[:,0]*sf, V_grid[:,1]*sf, scale=scale, scale_units='x', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid*sf, y_grid*sf, V_grid[0]*sf, V_grid[1]*sf, color=arrow_color, density=stream_density, linewidth=stream_linewidth)

    ax.axis("equal")
    ax.axis("off")
    if not filename is None:
        plt.savefig(filename, dpi=500, bbox_inches = 'tight', transparent=True)

    plt.show()


def plot_cell_adj(coord, cell_type, adj, filename=None, cmap=None, cell_size=25, connection_alpha=0.5):
    coord_r = coord.copy()
    # coord_r[:,1] = -coord_r[:,1]
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the axis limits
    ax.set_xlim(min(coord_r[:, 0]) - 10, max(coord_r[:, 0]) + 10)
    ax.set_ylim(min(coord_r[:, 1]) - 10, max(coord_r[:, 1]) + 10)

    # Plot cells
    sns.scatterplot(x=coord_r[:, 0], y=coord_r[:, 1], hue=cell_type, palette=cmap, edgecolor=None, ax=ax, s=cell_size)

    # Plot cell connections with varying linewidths
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] > 0:  # Check if there is a connection
                line_width = adj[i, j]  # Use adj value for line width
                line = Line2D(
                    [coord_r[i, 0], coord_r[j, 0]],  # x坐标
                    [coord_r[i, 1], coord_r[j, 1]],  # y坐标
                    color='black',
                    linewidth=line_width,
                    alpha=connection_alpha,
                )
                ax.add_line(line)

    # # Create legend
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

    # Remove axis border
    ax.axis('off')

    # Save the plot to a file
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.show()
    plt.close()


def plot_cell_connections(
    coord: np.ndarray,
    cell_type: list,
    adj: np.ndarray,
    directed: bool = False,
    sender: str = None,      # 新增参数
    receiver: str = None,    # 新增参数
    filename: str = None,
    cmap: dict = None,
    cell_size: int = 25,
    connection_alpha: float = 0.5,
    max_linewidth: float = 2.0,
    other_cell_color: str = '#D3D3D3' # 其他细胞的灰色
):
    """
    可视化细胞的空间位置及其通讯连接(支持聚焦于sender-receiver)。

    Args:
        coord (np.ndarray): 细胞坐标数组。
        cell_type (list): 每个细胞的类型列表。
        adj (np.ndarray): 邻接矩阵。
        directed (bool, optional): 是否绘制有向箭头。
        sender (str, optional): 指定的发送细胞类型。
        receiver (str, optional): 指定的接收细胞类型。
        filename (str, optional): 保存图像的文件路径。
        cmap (dict, optional): 细胞类型的颜色映射。
        cell_size (int, optional): 细胞点的大小。
        connection_alpha (float, optional): 连接线的透明度。
        max_linewidth (float, optional): 连接线的最大宽度。
        other_cell_color (str, optional): 非sender/receiver细胞的颜色。
    """
    if hasattr(adj, "toarray"):
        adj = adj.toarray()
        
    cell_type_array = np.array(cell_type)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # --- 1. 细胞点绘制逻辑 ---
    focus_mode = sender is not None and receiver is not None
    
    if focus_mode:
        # 聚焦模式：高亮sender/receiver，灰化其他细胞
        colors = []
        for ct in cell_type_array:
            if ct == sender:
                colors.append(cmap.get(sender, 'red')) # 如果cmap中没有，则用默认色
            elif ct == receiver:
                colors.append(cmap.get(receiver, 'blue'))
            else:
                colors.append(other_cell_color)
        
        sns.scatterplot(x=coord[:, 0], y=coord[:, 1], c=colors, edgecolor=None, linewidth=0.2, ax=ax, s=cell_size, zorder=2)
        
        # 创建自定义图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=sender, markerfacecolor=cmap.get(sender, 'red'), markersize=10),
            Line2D([0], [0], marker='o', color='w', label=receiver, markerfacecolor=cmap.get(receiver, 'blue'), markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=other_cell_color, markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    else:
        # 默认模式：绘制所有细胞类型
        sns.scatterplot(x=coord[:, 0], y=coord[:, 1], hue=cell_type, palette=cmap, edgecolor=None, linewidth=0.2, ax=ax, s=cell_size, zorder=2)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # --- 2. 连接绘制逻辑 ---
    adj_to_plot = adj.copy()
    if adj.max() > 0:
        adj_to_plot = adj_to_plot / adj.max() * max_linewidth
    
    if focus_mode:
        # 聚焦模式：过滤邻接矩阵，只保留 sender -> receiver 的连接
        sender_indices = np.where(cell_type_array == sender)[0]
        receiver_indices = np.where(cell_type_array == receiver)[0]
        
        mask = np.zeros_like(adj_to_plot)
        
        # 使用 np.ix_ 高效地创建掩码
        if len(sender_indices) > 0 and len(receiver_indices) > 0:
            mask[np.ix_(sender_indices, receiver_indices)] = 1
            # 对于无向图，也考虑 receiver -> sender 的连接
            if not directed:
                mask[np.ix_(receiver_indices, sender_indices)] = 1
        
        adj_to_plot *= mask # 应用掩码

    # --- 绘制连接（箭头或线段）---
    if directed:
        sources, targets = adj_to_plot.nonzero()
        for i, j in zip(sources, targets):
            start_pos, end_pos = coord[i], coord[j]
            line_width = adj_to_plot[i, j]
            ax.arrow(start_pos[0], start_pos[1], end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
                     color='black', alpha=connection_alpha, linewidth=line_width,
                     head_width=line_width * 2.5, head_length=line_width * 2.5,
                     length_includes_head=True, zorder=1)
    else:
        sources, targets = np.triu(adj_to_plot, k=1).nonzero()
        if len(sources) > 0:
            segments = np.array([coord[sources], coord[targets]]).transpose(1, 0, 2)
            linewidths = adj_to_plot[sources, targets]
            lc = LineCollection(segments, linewidths=linewidths, colors='black', alpha=connection_alpha, zorder=1)
            ax.add_collection(lc)

    # --- 图形美化 ---
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if focus_mode:
        ax.set_title(f"Connections from {sender} to {receiver}", fontsize=14)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()


def calculate_connection_matrix(adj, cell_type_label):
    # 将cell_type_label转换为一维数组
    cell_types = np.array(cell_type_label)

    # 获取独特的细胞类型
    unique_cell_types = np.unique(cell_types)
    num_types = len(unique_cell_types)

    # 创建连接数矩阵
    connection_matrix = np.zeros((num_types, num_types), dtype=float)

    # 创建细胞类型到索引的映射字典
    type_to_index = {cell_type: index for index, cell_type in enumerate(unique_cell_types)}

    # 遍历邻接矩阵的每一行
    adj = adj.toarray() if hasattr(adj, 'toarray') else adj
    for i in range(adj.shape[0]):
        # 获取该细胞的类型
        cell_type = cell_types[i]

        # 找到与该细胞相连的细胞的类型
        connected_idx = np.where(adj[i] > 0)[0]

        # 统计连接数
        for j in connected_idx:
            connection_matrix[type_to_index[cell_type], type_to_index[cell_types[j]]] += adj[i][j]

    # 创建连接数矩阵的数据框
    connection_df = pd.DataFrame(connection_matrix, index=unique_cell_types, columns=unique_cell_types)

    return connection_df


def interact_test(adjacency_matrix, cell_labels, num_permutations=1000):

    # Compute observed connection counts between each pair of cell types
    observed_counts = calculate_connection_matrix(adjacency_matrix,cell_labels)
    preserve_idx = (observed_counts!=0).any(axis=1)
    observed_counts = observed_counts.loc[preserve_idx,preserve_idx]
    
    unique_labels = observed_counts.index.to_numpy()
    num_labels = len(unique_labels)

    # Create dataframes to store p-values
    p_combined = observed_counts.copy()
    p_indicate = pd.DataFrame(index=unique_labels,columns=unique_labels)

    # Perform permutation tests
    for i in range(num_labels):
        for j in range(i, num_labels):

            observed_count = observed_counts.iloc[i, j]
            permuted_counts = np.zeros(num_permutations)

            for k in range(num_permutations):
                # Randomly permute cell labels
                permuted_labels = np.random.permutation(cell_labels)

                # Compute connection count between permuted cell types
                mask_i = (permuted_labels == unique_labels[i])
                mask_j = (permuted_labels == unique_labels[j])
                permuted_counts[k] = np.sum(adjacency_matrix[mask_i][:, mask_j])

            # Compute p-value for left-sided test
            p_value_left = np.sum(permuted_counts <= observed_count) / num_permutations

            # Compute p-value for right-sided test
            p_value_right = np.sum(permuted_counts >= observed_count) / num_permutations
            
            if p_value_left < p_value_right:
                p_combined.loc[unique_labels[i], unique_labels[j]] = p_combined.loc[unique_labels[j], unique_labels[i]] = p_value_left
                p_indicate.loc[unique_labels[i], unique_labels[j]] = p_indicate.loc[unique_labels[j], unique_labels[i]] = 'deplete'
            else:
                p_combined.loc[unique_labels[i], unique_labels[j]] = p_combined.loc[unique_labels[j], unique_labels[i]] = p_value_right
                p_indicate.loc[unique_labels[i], unique_labels[j]] = p_indicate.loc[unique_labels[j], unique_labels[i]] = 'enrich'

    return p_combined, p_indicate


def cal_range_proportion(lst, low_cut, high_cut):
    n = len(lst)
    if n==0:
        return None
    short_prop = np.sum(lst<low_cut)/n
    long_prop = np.sum(lst>high_cut)/n
    med_prop = 1-short_prop-long_prop
    return {'S':short_prop,'M':med_prop,'L':long_prop}


def classify_range(coord, adj, cell_types, q_short=30, q_long=70):

    dist_matrix = squareform(pdist(coord, 'euclidean'))
    adj_dist = np.triu(adj*dist_matrix)
    adj_dist = adj_dist[np.where(adj_dist!=0)]
    long_cutoff = np.percentile(adj_dist,q_long)
    short_cutoff = np.percentile(adj_dist,q_short)

    # 将cell_type_label转换为一维数组
    cell_types = np.array(cell_types)

    # 获取独特的细胞类型
    unique_cell_types = np.unique(cell_types)

    # 创建矩阵
    range_type_df = pd.DataFrame(index=unique_cell_types, columns=unique_cell_types)
    range_prop_df = pd.DataFrame(index=unique_cell_types, columns=unique_cell_types)
    range_componet_df = pd.DataFrame(index=unique_cell_types, columns=unique_cell_types)
    for tp1 in unique_cell_types:
        for tp2 in unique_cell_types:
            range_componet_df.loc[tp1,tp2] = []

    # 遍历邻接矩阵的每一行
    for i in range(adj.shape[0]):
        # 获取该细胞的类型
        cell_type = cell_types[i]

        # 找到与该细胞相连的细胞的类型和对应距离
        connect_idx = np.where(adj[i] == 1)
        connected_cell_types = cell_types[connect_idx]
        connected_dists = dist_matrix[i][connect_idx]

        # 收集距离分布
        for j,ct in enumerate(connected_cell_types):
            range_componet_df.loc[cell_type,ct].append(connected_dists[j])

    for tp1 in unique_cell_types:
        for tp2 in unique_cell_types:
            tmp = cal_range_proportion(range_componet_df.loc[tp1,tp2],short_cutoff,long_cutoff)
            if tmp is not None:
                range_prop_df.loc[tp1,tp2] = list(tmp.values())
                range_type_df.loc[tp1,tp2] = max(tmp.items(), key=lambda x: x[1])[0]

    return range_type_df, range_prop_df, range_componet_df


def classify_range2(coord, adj, devide_q=50):

    dist_matrix = squareform(pdist(coord, 'euclidean'))
    adj_dist = np.triu(adj*dist_matrix)
    adj_dist = adj_dist[np.where(adj_dist!=0)]
    cutoff = np.percentile(adj_dist,devide_q)

    adj_short = adj.copy()
    adj_long = adj.copy()
    adj_short[np.where(dist_matrix>cutoff)] = 0
    adj_long[np.where(dist_matrix<=cutoff)] = 0

    return adj_short, adj_long


def celltype_adj(used_ct, adj, all_cts):

    cell_types = np.array(all_cts)
    remove_idx = np.where(cell_types!=used_ct)[0]
    ct_adj = adj.copy()
    ct_adj[np.ix_(remove_idx, remove_idx)] = 0

    return ct_adj


def random_edge(adj, frac=0.001):

    edge_idx = np.where(adj==1)
    fill = np.random.choice([0,1],size=edge_idx[0].shape,replace=True,p=[1-frac,frac])
    select_adj = adj.copy()
    select_adj[edge_idx] = fill

    return select_adj