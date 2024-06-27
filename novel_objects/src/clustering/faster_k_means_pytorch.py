import numpy as np
import copy
import random
# from project_utils.cluster_utils import cluster_acc
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pairwise_distance(data1, data2, batch_size=None):
    # N*1*M
    A = torch.from_numpy(np.expand_dims(data1, axis=1))
    # 1xNxM
    B = data2.unsqueeze(dim=0)

    if batch_size==None:
        # A = data1.unsqueeze(dim=1)
        # B = data2.unsqueeze(dim=0)

        dis = (A - B) ** 2
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if (i + batch_size < data1.shape[0]):
                dis_batch = (A[i:i + batch_size] - B) ** 2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i + batch_size] = dis_batch
                i = i + batch_size
                #  torch.cuda.empty_cache()
            elif (i + batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B) ** 2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
        #  torch.cuda.empty_cache()
    return dis


class K_Means:

    def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
                 n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode

    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)

        if pre_centers is not None:
            C = pre_centers
        else:
            C = X[random_state.randint(0, len(X))]

        C = torch.from_numpy(C)
        C = C.view(-1, X.shape[-1])

        while C.shape[0] < k:

            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2 / d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, torch.from_numpy(X[ind]).view(1, -1)), dim=0)

        return C

    def fit_once(self, X, random_state):

        centers = torch.zeros(self.k, X.shape[-1])
        labels = -torch.ones(len(X))

        # initialize the centers!

        if self.init == 'k-means++':
            centers = self.kpp(X, k=self.k, random_state=random_state)

        elif self.init == 'random':

            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]

        # begin iterations

        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):

            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            min_dist, labels = torch.min(dist, dim=1)
            inertia = min_dist.sum()

            for idx in range(self.k):
                selected = torch.nonzero(labels==idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                break

        return best_labels, best_inertia, best_centers, i+1

    def visualise(X, kmeans):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            all_label = np.Inf * np.ones((len(kmeans.cluster_centers_), len(X)))
            for i, feat in enumerate(X):
                print(i)
                for j, mean in enumerate(kmeans.cluster_centers_):
                    all_label[j, i] = np.sqrt(np.sum(np.square(mean-feat)))
            min_cluster_labels = np.argmin(all_label, 0)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X)
            df = pd.DataFrame()
            df['pca-one'] = pca_result[:, 0]
            df['pca-two'] = pca_result[:, 1]
            df['y'] = [np.uint8(i) for i in min_cluster_labels]
            df.to_csv("load.csv")

        plt.figure(figsize=(16, 10))
        customPalette = sns.set_palette(sns.color_palette("husl", 25))
        hexpalette = ['#970DCF', '#4D5E07', '#608AD6', '#632D3B', '#FC2334',  '#C5498A', '#87CCB0', '#E6CB7F',
            '#E0C0DD', '#96141E', '#3FEDF6', '#AB4723', '#1AE01B', '#69C5E9', '#71774E', '#6B0C5A', '#154075', '#DA8B1F',
                      '#402092', '#FEAE94', '#A5E733', '#B9109C', '#707FF1', '#8BBDC4', '#FD6861']
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            data=df,
            palette=hexpalette,
            legend="full",
            alpha=0.3
        )
        plt.savefig("kmeans_hexpalette.png")



    def fit_cosine_kmeans(self, X, random_state):

        kmeans = KMeans(n_clusters=self.k, n_init=self.n_init, max_iter=self.max_iterations, init = 'k-means++',
                                tol=self.tolerance, random_state=random_state, verbose=1).fit(X)

        # i = 0
        # batch_size = self.pairwise_batch_size
        # label_list = []
        # while i < X.shape[0]:
        #     if i == 0:
        #         A = X[i:i + batch_size]
        #         i = i + batch_size
        #         kmeans = KMeans(n_clusters=self.k, n_init=self.n_init, max_iter=self.max_iterations, init = 'k-means++',
        #                         tol=self.tolerance, random_state=random_state, verbose=False).fit(A)
        #         label_list.append(kmeans.labels_)
        #     elif (i + batch_size < X.shape[0]):
        #         A = X[i:i+batch_size]
        #         i = i + batch_size
        #         kmeans = KMeans(n_clusters=self.k, n_init=1, max_iter=self.max_iterations, init=init,
        #                         tol=self.tolerance, random_state=0, verbose=False).fit(A)
        #         label_list.append(kmeans.labels_)
        #     elif (i + batch_size >= X.shape[0]):
        #         A = X[i:]
        #         kmeans = KMeans(n_clusters=self.k, n_init=1, max_iter=self.max_iterations, init=init,
        #                 tol=self.tolerance, random_state=random_state, verbose=True).fit(A)
        #         label_list.append(kmeans.labels_)
        #         break
        #     init = kmeans.cluster_centers_
            # import os
            # if os.path.isfile( "./load.csv"):
            #     df = pd.read_csv("load.csv", header=0, index_col=0)
            #     self.visualise(df, kmeans)
            # else:
            #     self.visualise(X, kmeans)

        # self.visualise_features(X, kmeans)
        return kmeans.labels_, kmeans.inertia_, kmeans.cluster_centers_, kmeans.n_iter_

    def fit(self, X):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            labels, inertia, centers, n_iters = self.fit_cosine_kmeans(X, random_state)
            # for it in range(self.n_init):
            #     # labels, inertia, centers, n_iters = self.fit_once(X, random_state)
            #     labels, inertia, centers, n_iters = self.fit_cosine_kmeans(X, random_state)
            #     if best_inertia is None or inertia < best_inertia:
            self.labels_ = labels
            self.cluster_centers_ = centers
            self.inertia_ = inertia
            self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed)
                                                              for seed in seeds)
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]

def main():
    import matplotlib.pyplot as plt
    from matplotlib import style
    import pandas as pd
    style.use('ggplot')
    from sklearn.datasets import make_blobs
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility

    cuda = torch.cuda.is_available()
    #  X = torch.from_numpy(X).float().to(device)
    y = np.array(y)
    targets = y[y>1]
    feats = X[y>1]
    feats = torch.from_numpy(feats).to(device)
    targets = torch.from_numpy(targets).to(device)

