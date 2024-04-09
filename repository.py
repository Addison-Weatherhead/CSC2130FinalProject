

class Repository:
    def __init__(self, name, n_contributors, n_stars, n_openissues, n_PRs):
        self.name = name
        self.workflows = {} # a dict of the contents of yaml files. Empty if the repo has no workflows. Values are 'run' objects from that workflow
        self.n_contributors = n_contributors
        self.n_stars = n_stars
        self.n_openissues = n_openissues
        self.n_PRs = n_PRs

        
