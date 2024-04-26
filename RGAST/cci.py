import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.collections import LineCollection

def attention_to_interact(attention, n_spots, cum_att_threshold=0.5):
    interact_mat = np.zeros((n_spots,n_spots))
    for i in np.arange(n_spots):
        idx = np.where(attention[0][1]==i)[0]
        from_idx = attention[0][0,idx]
        att_score = attention[1][np.where(attention[0][1]==i)].reshape(-1)
        sorted_indices = np.argsort(att_score)[::-1]
        cumulative_sum = np.cumsum(att_score[sorted_indices])
        stop_index = np.where(cumulative_sum > cum_att_threshold)[0][0]
        used_from_idx = from_idx[sorted_indices[:stop_index+1]]
        interact_mat[i,used_from_idx] = interact_mat[used_from_idx,i] = 1
    return interact_mat

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

def plot_cell_adj(coord, cell_type, adj, filename=None, cmap=None, cell_size=25):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the axis limits
    ax.set_xlim(min(coord[:, 0]) - 1, max(coord[:, 0]) + 1)
    ax.set_ylim(min(coord[:, 1]) - 1, max(coord[:, 1]) + 1)

    # Plot cells
    sns.scatterplot(x=coord[:, 0], y=coord[:, 1], hue=cell_type, palette=cmap, edgecolor=None, ax=ax, s=cell_size)

    # Plot cell connections
    lines = []
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            if adj[i, j] == 1:
                lines.append([coord[i], coord[j]])

    line_collection = LineCollection(lines, linewidths=1, colors='black', alpha=0.5)
    ax.add_collection(line_collection)

    # # Create legend
    # legend_labels = np.unique(cell_type)
    # legend_handles = scatter_handles[:len(legend_labels)]
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

    # Remove axis border
    ax.axis('off')

    # Save the plot to a file
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        
    plt.close()


def calculate_connection_matrix(adj, cell_type_label):
    # 将cell_type_label转换为一维数组
    cell_types = np.array(cell_type_label)

    # 获取独特的细胞类型
    unique_cell_types = np.unique(cell_types)
    num_types = len(unique_cell_types)

    # 创建连接数矩阵
    connection_matrix = np.zeros((num_types, num_types), dtype=int)

    # 创建细胞类型到索引的映射字典
    type_to_index = {cell_type: index for index, cell_type in enumerate(unique_cell_types)}

    # 遍历邻接矩阵的每一行
    for i in range(adj.shape[0]):
        # 获取该细胞的类型
        cell_type = cell_types[i]

        # 找到与该细胞相连的细胞的类型
        connected_cell_types = cell_types[np.where(adj[i] == 1)]

        # 统计连接数
        for j in connected_cell_types:
            connection_matrix[type_to_index[cell_type], type_to_index[j]] += 1

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