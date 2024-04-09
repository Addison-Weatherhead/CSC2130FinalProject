import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, sm
from statsmodels.stats.anova import anova_lm

def multivariatelinearRegression(REPO):
    data = {
        'n_contributors': [repo.n_contributors for repo in REPO],
        'n_stars': [repo.n_stars for repo in REPO],
        'n_openissues': [repo.n_openissues for repo in REPO],
        'n_PRs': [repo.n_PRs for repo in REPO],
        'has_workflow': [1 if repo.workflows else 0 for repo in REPO]  # 1 if workflows is not empty, else 0
    }
    df = pd.DataFrame(data)
    
    X = df[['n_contributors', 'n_openissues', 'n_PRs', 'has_workflow']]  # independent variables
    y = df['n_stars']  # dependent variable

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Summary of the model
    print(model.summary())
    
def anova(REPO):
    data = {
        'n_contributors': [repo.n_contributors for repo in REPO],
        'n_stars': [repo.n_stars for repo in REPO],
        'n_openissues': [repo.n_openissues for repo in REPO],
        'n_PRs': [repo.n_PRs for repo in REPO],
        'has_workflow': [1 if repo.workflows else 0 for repo in REPO]  # 1 if workflows is not empty, else 0
    }
    df = pd.DataFrame(data)
    
    model = ols('n_contributors ~ C(has_workflow)', data=df).fit()
    anova_results = anova_lm(model)
    
    
    print(anova_results)
    
def kmeans(REPO, k_values):
    data = {
        'n_contributors': [repo.n_contributors for repo in REPO],
        'n_stars': [repo.n_stars for repo in REPO],
        'n_openissues': [repo.n_openissues for repo in REPO],
        'n_PRs': [repo.n_PRs for repo in REPO],
        'has_workflow': [1 if repo.workflows else 0 for repo in REPO]  # 1 if workflows is not empty, else 0
    }
    df = pd.DataFrame(data)
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)

    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42).fit(df_normalized)
        inertias.append(model.inertia_)

    # Plotting the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, '-o')
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def PCA(REPO):
    data = {Remodified query
        'n_contributors': [repo.n_contributors for repo in REPO],
        'n_stars': [repo.n_stars for repo in REPO],
        'n_openissues': [repo.n_openissues for repo in REPO],
        'n_PRs': [repo.n_PRs for repo in REPO],
        'has_workflow': [1 if repo.workflows else 0 for repo in REPO]  # 1 if workflows is not empty, else 0
    }
    df = pd.DataFrame(data)
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    
    pca = PCA().fit(df_normalized)

    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') # for each component
    plt.title('Explained Variance by Components')
    plt.xticks(range(0, len(data.keys()), 1))
    plt.grid(True)
    plt.show()