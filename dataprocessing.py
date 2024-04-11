import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

def multivariatelinearRegression(REPOS, ind_vars, dep_var):
    df = preprocessing(REPOS)
    
    X = df[ind_vars]  # independent variables
    y = df[dep_var]  # dependent variable

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Summary of the model
    print(model.summary())
    
def anova(REPOS):
    df = preprocessing(REPOS)
    
    model = ols('n_contributors ~ C(has_workflow)', data=df).fit()
    anova_results = anova_lm(model)
    
    
    print(anova_results)
    
def kmeans(REPOS, k_values):
    df = preprocessing(REPOS, normalize = True)

    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42).fit(df)
        inertias.append(model.inertia_)

    # Plotting the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, '-o')
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def PCA(REPOS):
    df = preprocessing(REPOS, normalize = True)
    
    pca = PCA().fit(df)

    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') # for each component
    plt.title('Explained Variance by Components')
    plt.xticks(range(0, len(data.keys()), 1))
    plt.grid(True)
    plt.show()
    

def preprocessing(REPOS, normalize = False):
    data = {
        'n_contributors': [repo.n_contributors for repo in REPOS],
        'n_stars': [repo.n_stars for repo in REPOS],
        'n_openissues': [repo.n_openissues for repo in REPOS],
        'n_PRs': [repo.n_PRs for repo in REPOS],
        'n_workflows': [repo.n_workflows for repo in REPOS],
        'n_workflowruns': [repo.n_workflowruns for repo in REPOS],
        'issues_per_contributor': [0 if repo.n_contributors==0 else repo.n_openissues/repo.n_contributors for repo in REPOS],
    }
    df = pd.DataFrame(data)
    if normalize:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    
    return df