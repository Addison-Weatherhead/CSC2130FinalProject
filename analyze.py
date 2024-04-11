from dataprocessing import *
import pickle

REPOS = pickle.load(open("data.pkl", "rb"))
multivariatelinearRegression(REPOS, ['n_workflows', 'n_workflowruns', 'n_contributors', 'n_openissues', 'n_PRs'], 'n_stars')
multivariatelinearRegression(REPOS, ['n_contributors', 'n_openissues', 'n_PRs'], 'n_stars')

print('\n\n\n=====================\n\n\n')

multivariatelinearRegression(REPOS, ['n_workflows', 'n_workflowruns', 'n_contributors', 'n_PRs', 'n_stars'], 'n_openissues')
multivariatelinearRegression(REPOS, ['n_contributors', 'n_PRs', 'n_stars'], 'n_openissues')


print('\n\n\n=====================\n\n\n')

multivariatelinearRegression(REPOS, ['n_workflows', 'n_workflowruns', 'n_contributors', 'n_PRs', 'n_stars'], 'issues_per_contributor')
multivariatelinearRegression(REPOS, ['n_contributors', 'n_PRs', 'n_stars'], 'issues_per_contributor')


n_with_workflows = len([repo for repo in REPOS if repo.n_workflows > 0])
print('n_with_workflows: ', n_with_workflows)


# Pearson correlation between n_workflows and n_stars
n_workflows = [repo.n_workflows for repo in REPOS]
n_stars = [repo.n_stars for repo in REPOS]
n_contributors = [repo.n_contributors for repo in REPOS]

from scipy.stats import pearsonr
correlation, _ = pearsonr(n_workflows, n_stars)
print('correlation with n_workflows and n_stars: ', correlation)

correlation, _ = pearsonr(n_workflows, n_contributors)
print('correlation with n_workflows and n_contributors: ', correlation)

