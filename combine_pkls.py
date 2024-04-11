import pickle
import os
from repository import Repository
repo_names = []
REPOS = []

for filename in os.listdir('data/'):
    if filename.endswith('.pkl'):
        filepath = os.path.join('data/', filename)
        print('processing: ', filepath)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            print('len: ', len(data))
            for repo in data:
                if repo.name not in repo_names:
                    repo_names.append(repo.name)
                    REPOS.append(repo)

print('len(REPOS): ', len(REPOS))
with open ("data.pkl", "wb") as f:
    pickle.dump(REPOS, f)
        