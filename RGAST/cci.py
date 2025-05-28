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

    S_np = adata.obsp['weighted_mat']
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