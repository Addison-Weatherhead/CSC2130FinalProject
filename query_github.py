from github import Github
from repository import Repository
import pickle

def create_data():
    token = open("githubtoken.txt", "r").read().strip()
    g = Github(token)

    rate_limit = g.get_rate_limit()
    print("Remaining API requests this hour: ", rate_limit.search.remaining)


    query = "machine_learning+machine-learning+ML+AI+Machine Learning+pytorch+tensorflow+pytorch-lightning+pytorchlightning+pytorch_lightning in:name,readme"
    result = g.search_repositories(query=query)

    REPOS = []

    for repo in result:
        if len(REPOS) % 100 == 0:
            print('Num processed: ', len(REPOS))
        repository = Repository(repo.name, repo.get_contributors().totalCount, repo.stargazers_count, repo.open_issues_count, repo.get_pulls().totalCount)

        workflows = repo.get_workflows()
        for workflow in workflows:
            # Fetching workflow runs using the workflow ID
            workflow_runs = []
            runs = workflow.get_runs()  # Using get_runs() method of the workflow object
            for run in runs:
                workflow_runs.append({
                    "id": run.id,
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "created_at": run.created_at,
                })

            if workflow_runs:  # If we have runs, fetch the YAML content
                try:
                    content = repo.get_contents(workflow.path)
                    yaml_content = content.decoded_content.decode("utf-8")
                    repository.workflows[yaml_content] = workflow_runs
                except Exception as e:
                    print(f"Could not access workflow file {workflow.path} for {repo.name}: {e}")

        REPOS.append(repository)
    g.close()

    with open ("data.pkl", "wb") as f:
        pickle.dump(REPOS, f)
    return REPOS