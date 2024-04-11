from dataprocessing import *
import pickle
import seaborn as sns

def comparitive_MV_Regression(REPOS, all_vars, target_var, remove_vars):
    ''' Compare LR performance between a model trained on all_vars-target_metric to predict target_metric
    vs a model trained on all_vars-target_var-remove_vars to predict target_metric
    '''
    print('=='*15)
    print('Model Trained with ', [var for var in all_vars if var != target_var], ' to predict ', target_var)
    multivariatelinearRegression(REPOS, [var for var in all_vars if var != target_var], target_var)
    print('\n'*4)
    print('Model Trained with ', [var for var in all_vars if var != target_var and var not in remove_vars], ' to predict ', target_var)
    multivariatelinearRegression(REPOS, [var for var in all_vars if var != target_var and var not in remove_vars], target_var)
    print('=='*15)
def main():
    REPOS = pickle.load(open("data.pkl", "rb"))
    print('len(REPOS): ', len(REPOS))
    print('Proportino of REPOS with >0 workflow: ', len([repo for repo in REPOS if repo.n_workflows>0])/len(REPOS))
    all_vars = ['n_workflows', 'n_workflowruns', 'n_contributors', 'n_stars', 
                'n_openissues', 'n_PRs', 'n_train_workflows', 'n_test_workflows', 
                'n_preprocess_workflows', 'n_clean_data_workflows']


    #comparitive_MV_Regression(REPOS, all_vars, target_var='n_openissues', remove_vars=['n_workflows', 'n_workflowruns'])
    #comparitive_MV_Regression(REPOS, all_vars, target_var='issues_per_contributor', remove_vars=['n_workflows', 'n_workflowruns'])
    #comparitive_MV_Regression(REPOS, all_vars, target_var='n_PRs', remove_vars=['n_workflows', 'n_workflowruns'])
    #comparitive_MV_Regression(REPOS, all_vars, target_var='n_stars', remove_vars=['n_workflows', 'n_workflowruns'])



    total_train = total_test = total_preprocess = total_clean_data = 0
    for repo in REPOS:
        total_train += repo.n_train_workflows
        total_test += repo.n_test_workflows
        total_preprocess += repo.n_preprocess_workflows
        total_clean_data += repo.n_clean_data_workflows
    
    workflow_types = ['Train', 'Test', 'Preprocess', 'Clean Data']
    workflow_counts = [total_train, total_test, total_preprocess, total_clean_data]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(workflow_types, workflow_counts, color='blue')
    plt.xlabel('Type of Workflow')
    plt.ylabel('Total Workflows')
    plt.title('Total Workflows by Type Across Repositories')
    plt.savefig('plots/type_of_workflow_hist.png')




    openissues_zero_workflows = [repo.n_openissues for repo in REPOS if repo.n_workflows == 0]
    openissues_nonzero_workflows = [repo.n_openissues for repo in REPOS if repo.n_workflows > 0]

    plt.figure(figsize=(12, 6))
    plt.hist(openissues_zero_workflows, color='blue', alpha=0.7, label='n_workflows=0', bins=range(0, 12, 1), density=True)
    plt.hist(openissues_nonzero_workflows, color='green', alpha=0.7, label='n_workflows>0', bins=range(0, 12, 1), density=True)

    plt.title('Normalized Comparison of # of Open Issues Distribution')
    plt.xlabel('Number of Open Issues')
    plt.ylabel('Probability Density')
    plt.legend()

    plt.savefig('plots/issues_givenworkflows_vs_noworkflows.png')





    PRs_zero_workflows = [repo.n_PRs for repo in REPOS if repo.n_workflows == 0]
    PRs_nonzero_workflows = [repo.n_PRs for repo in REPOS if repo.n_workflows > 0]

    plt.figure(figsize=(12, 6))
    plt.hist(PRs_zero_workflows, color='blue', alpha=0.7, label='n_workflows=0', bins=range(0, 12, 1), density=True)
    plt.hist(PRs_nonzero_workflows, color='green', alpha=0.7, label='n_workflows>0', bins=range(0, 12, 1), density=True)

    plt.title('Normalized Comparison of # of PRs Distributions')
    plt.xlabel('Number of PRs')
    plt.ylabel('Probability Density')
    plt.legend()

    plt.savefig('plots/PRs_givenworkflows_vs_noworkflows.png')






    df = preprocessing(REPOS)[['n_workflows', 'n_contributors', 'n_stars', 
                'n_openissues', 'n_PRs', 'n_train_workflows', 'n_test_workflows']]
    correlation_matrix = df.corr() # does pearson by default
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5)
    plt.xticks(rotation=45, ha='right')
    plt.title('(Pearson) Correlation Matrix of Repository Attributes')
    plt.tight_layout()

    plt.savefig('plots/CorrelationMatrix.png')








if __name__ == '__main__':
    main()

