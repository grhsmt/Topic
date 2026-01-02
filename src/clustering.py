from sklearn.cluster import KMeans, SpectralClustering, Birch, AgglomerativeClustering

def hierarchical_cluster(input_matrix, n_clusters, linkage='ward'):
    """
    层次聚类
    
    :param linkage: 连接方式 ('ward', 'complete', 'average', 'single')
    :return: (model, cluster_labels)
    """
    # 注意：ward 需要欧氏距离，cosine距离需用其他linkage
    model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage,
        metric='cosine' if linkage != 'ward' else 'euclidean'
    )
    labels = model.fit_predict(input_matrix.toarray() if linkage == 'ward' else input_matrix)
    return model, labels

def birch_cluster(input_matrix, n_clusters, threshold=0.5):
    """
    BIRCH 层次聚类
    
    :param threshold: 簇半径阈值
    :return: (model, cluster_labels)
    """
    model = Birch(n_clusters=n_clusters, threshold=threshold)
    labels = model.fit_predict(input_matrix)
    return model, labels

def spectral_cluster(input_matrix, n_clusters, affinity='rbf'):
    """
    谱聚类
    
    :param affinity: 相似度度量 ('rbf', 'nearest_neighbors', 'precomputed')
    :return: (model, cluster_labels)
    """
    model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
    labels = model.fit_predict(input_matrix)
    return model, labels

def kmeans_cluster(input_matrix, n_clusters):
    """
    KMeans 聚类
    
    :param n_clusters: 聚类数量
    :return: (model, cluster_labels)
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(input_matrix)
    return model, labels