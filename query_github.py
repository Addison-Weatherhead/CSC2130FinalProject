from github import Github
import github
from repository import Repository
import pickle
import time
import base64

def create_data():
    token = open("githubtoken.txt", "r").read().strip()
    g = Github(token)

    rate_limit = g.get_rate_limit()
    print("Remaining API requests this hour: ", rate_limit.search.remaining)


    REPOS = []
    repo_names = []
    
    #for query_str in ['Machine Learning', 'machine_learning', 'machine-learning', 'ML', 'AI']:
    for query_str in ['ML', 'AI']:
        print('query_str: ', query_str)
        query = query_str+" in:name,readme NOT course NOT tutorial"
        result = g.search_repositories(query=query)

        for repo in result:
            if repo.name in repo_names:
                continue
            rate_limit = g.get_rate_limit().core
            if rate_limit.remaining < 5:
                reset_time = rate_limit.reset.timestamp()  # Get the timestamp when the limit resets
                current_time = time.time()
                sleep_time = reset_time - current_time + 10  # Adding a 10-second buffer to be safe
                if sleep_time > 0:
                    print(f"Approaching rate limit. Sleeping for {sleep_time} seconds.")
                    time.sleep(sleep_time)  # Sleep until the rate limit resets
            
            if len(REPOS) % 100 == 0:
                print('Num processed: ', len(REPOS))
                with open ("data/%s.pkl"%query_str, "wb") as f: # checkpoint
                    pickle.dump(REPOS, f)
            repository = Repository(repo.name, repo.get_contributors().totalCount, repo.stargazers_count, repo.open_issues_count, repo.get_pulls().totalCount)

            try: 
                workflows = repo.get_workflows()
                num_workflows = workflows.totalCount
                repository.n_workflows = num_workflows
                
                for file in repo.get_contents(".github/workflows"):
                    if file.name.endswith(".yml") or file.name.endswith(".yaml"):
                        file_content = base64.b64decode(file.content).decode('utf-8')
                        if 'train' in file_content:
                            repository.n_train_workflows += 1
                        elif 'test' in file_content:
                            repository.n_test_workflows += 1
                        elif 'preprocess' in file_content:
                            repository.n_preprocess_workflows += 1
                        elif 'clean-data' in file_content:
                            repository.n_clean_data_workflows += 1
                        elif 'clean_data' in file_content:
                            repository.n_clean_data_workflows += 1

            except github.GithubException as e:
                pass
            REPOS.append(repository)
            repo_names.append(repo.name)


        with open ("data/%s.pkl"%query_str, "wb") as f:
            pickle.dump(REPOS, f)
        return REPOS

    g.close()