

class Repository:
    def __init__(self, name, n_contributors, n_stars, n_openissues, n_PRs):
        self.name = name
        self.n_workflows = 0
        self.n_workflowruns = 0
        self.n_contributors = n_contributors
        self.n_stars = n_stars
        self.n_openissues = n_openissues
        self.n_PRs = n_PRs
        self.n_train_workflows = 0
        self.n_test_workflows = 0
        self.n_preprocess_workflows = 0
        self.n_clean_data_workflows = 0

        
