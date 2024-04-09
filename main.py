from github import Github
from repository import Repository
from query_github import create_data
import os
import pickle
import time

def main():
    if not os.path.exists("data.pkl"):  
        print('Querying data... (current time: ', time.ctime(), ')')
        REPOS = create_data()
    else:
        print('Loading saved data... (current time: ', time.ctime(), ')')
        REPOS = pickle.load(open("data.pkl", "rb"))
    print('Data Loaded! (current time: ', time.ctime(), ')')
    print('Number of repos found: ', len(REPOS))
if __name__ == "__main__":
    main()