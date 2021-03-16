References:
    


    External data taken from https://www.reddit.com/r/datasets/comments/6fniik/over_one_million_tweets_collected_from_us/    
    Embedding Algorithm: https://github.com/stanfordnlp/GloVe    
    Raw code with the embedding training can be found on https://purdue0-my.sharepoint.com/:f:/g/personal/skodge_purdue_edu/EhJaVG4if69IjIOuCTwo0LMBsH_za6tTpAw6pr599HRI9Q?e=GgxYkF

Running the codes

Preprocessing the online tweet data data(assuming the directory is downloaded form the link https://purdue0-my.sharepoint.com/:f:/g/personal/skodge_purdue_edu/EhJaVG4if69IjIOuCTwo0LMBsH_za6tTpAw6pr599HRI9Q?e=GgxYkF)



    cd hw1/
    python corpus_creation.py

Embedding:(assuming the directory is downloaded form the link https://purdue0-my.sharepoint.com/:f:/g/personal/skodge_purdue_edu/EhJaVG4if69IjIOuCTwo0LMBsH_za6tTpAw6pr599HRI9Q?e=GgxYkF)



    cd hw1/glove
    sh demo.sh

Training:(See the training file for changing the arguments. To run it for the first time pass "-cv True" This will save the embedding vector as a np array)



    cd hw1/
    python main.py 
