"""
Find all the details on https://github.com/Sangameshkodge/CS-57700.git
I use glove(https://github.com/stanfordnlp/GloVe) for generating word embeddings from dataset found online (https://www.reddit.com/r/datasets/comments/6fniik/over_one_million_tweets_collected_from_us/).
The raw directory found on (https://purdue0-my.sharepoint.com/:f:/g/personal/skodge_purdue_edu/EhJaVG4if69IjIOuCTwo0LMBsH_za6tTpAw6pr599HRI9Q?e=GgxYkF) 
Author: Sangamesh Kodge
email: skodge
"""
import csv
from random import randrange, shuffle
from nltk.tokenize import TweetTokenizer
import numpy as np
import pickle 
#from tqdm import tqdm
import matplotlib.pyplot as plt

### Creates a vocabs
class small_vocab():
    def __init__(self, tweet_id2text, tweet_id2issue, tweet_id2author_label, vocab_size=None, embed = None, combined_vocab = None ):
        self.vocab_size = vocab_size
        self.issue2index, self.issue2word, self.issue_freq, self.len_issue = self.issue_vocab(tweet_id2issue)
        self.author_label2index, self.index2author_label, self.author_label_freq, self.len_author = self.author_label_vocab(tweet_id2author_label)
        self.sample_size = len(tweet_id2text) 
        self.embed= embed
        self.word2index, self.index2word, self.word_freq, self.len_vocab, self.max_len = self.make_vocab(tweet_id2text)
        if embed is not None:
            self.vocab_size = embed.shape[0]
            self.embed_size = embed.shape[1]
            self.combined_vocab = combined_vocab            
    def make_vocab(self, tweet_id2text):
        ###Count the frequency of the word
        tokenizer = TweetTokenizer()
        word_freq = {}
        max_len = 0 
        for tweet_id in tweet_id2text:
            word_list = tokenizer.tokenize(tweet_id2text[tweet_id])
            if max_len < len(word_list):
                max_len = len(word_list)
            for word in word_list:
                if word in word_freq:
                    word_freq[word]+=1
                else:
                    word_freq[word]=1        
        ###order the words according to frequency. 
        ordered_vocab = sorted(word_freq.items() , key=lambda t : t[1] , reverse=True)
        if self.vocab_size is not None:
            len_vocab = min(len(ordered_vocab), self.vocab_size)
        else:
            len_vocab = len(ordered_vocab)        
        ###create a dict for word to index and index to word
        word2index = {}
        index2word = {}
        wordfreq = {}
        for count,(word, freq) in enumerate(ordered_vocab):
            word2index[word] = count
            index2word[count] = word
            wordfreq [count] = freq
            if self.vocab_size is not None:
                if count >= self.vocab_size:
                    break
        return word2index, index2word, wordfreq, len_vocab, max_len
    def issue_vocab(self, tweet_id2issue):
        ###counts the frequency of issue and assigns it an integer.
        issue_freq = {}
        for tweet_id in tweet_id2issue:
            issue = tweet_id2issue[tweet_id].strip()
            if issue in issue_freq:
                issue_freq[issue] +=1
            else:
                issue_freq[issue] = 1
        ###order the issue according to frequency. 
        ordered_issue = sorted(issue_freq.items() , key=lambda t : t[1] , reverse=True)
        len_issue=len(ordered_issue)
        
        ###create a dict for issue to index and index to isue and index to freq
        issue2index = {}
        index2issue = {}
        for count,(issue, freq) in enumerate(ordered_issue):
            issue2index[issue] = count
            index2issue[count] = issue
            issue_freq[count] = freq
        return issue2index, index2issue, issue_freq, len_issue
    def author_label_vocab(self, tweet_id2author_label):
        ###
        author_label_freq = {}
        for tweet_id in tweet_id2author_label:
            author = tweet_id2author_label[tweet_id].strip()
            if author in author_label_freq:
                author_label_freq[author] +=1
            else:
                author_label_freq[author] = 1
        ##order the author according to frequency. 
        ordered_author_label = sorted(author_label_freq.items() , key=lambda t : t[1] , reverse=True)
        len_author=len(author_label_freq)        
        ###create a dict for author to index and index to author and index to freq
        author_label2index = {}
        index2author_label = {}
        for count,(author, freq) in enumerate(ordered_author_label):
            author_label2index[author] = count
            index2author_label[count] = author
            author_label_freq[count] = freq
        return author_label2index, index2author_label, author_label_freq, len_author

class glove_vocab():
    def __init__(self, embed_size=64):
        self.word2index={}
        self.index2word={}
        self.index2freq = {}
        count = 0
        with open("./glove/vocab.txt", "r") as f:
            for line in f:
                line = line[:-1].split(" ")
                self.word2index[line[0]] = count
                self.index2word[count] = line[0]
                self.index2freq[count] = int(line[1])
                count+=1
        self.word2index["<unk>"] = count
        self.index2word[count] = "<unk>"
        self.index2freq[count] = None
        self.word2index[line[0]] = count+1
        self.index2word[count+1] = "<pad>"
        self.index2freq[count+1] = None
        self.vocab_size = count+2 
        self.embed_len = embed_size
        self.embed_matrix = np.zeros((self.vocab_size,self.embed_len))        
        with open("./glove/vectors.txt", "r") as f:
            for line in f:
                line = line.strip()[:-1].split(" ")
                if line[0] in self.word2index:
                    self.embed_matrix[self.word2index[line[0]]] = np.array([float(val) for val in line[1:] ])
                else:
                    if line[0] == "<unk>":
                        self.embed_matrix[-1] = np.array([float(val) for val in line[1:] ])
                    else:
                        self.embed_matrix[-2] = np.array([float(val) for val in line[0:] ])

#### Creates DATASET
class dataset():
    def __init__(self, tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, vocab):
        self.tweet_id2text = tweet_id2text
        self.tweet_id2issue = tweet_id2issue
        self.tweet_id2author_label = tweet_id2author_label
        self.tweet_id2label = tweet_id2label
        self.vocab = vocab
        self.author_label2index = self.author_label_formating()
        self.issue2index = self.issue_formating()
        self.text2vec = self.text_formating()    
        self.label2index = self.label_formating()
        self.data = self.vectorize()
        shuffle(self.data)
    def label_formating(self):
        label2index = {}
        for tweet_id in self.tweet_id2label:
            if self.tweet_id2label[tweet_id] != "None":
                label2index[tweet_id]= int( self.tweet_id2label[tweet_id]) 
            else:
                label2index[tweet_id]=None

        return label2index
    def author_label_formating(self):
        ###returns 0 for republican and 1 for democrat else returns true
        author_label2index = {}
        for tweet_id in self.tweet_id2author_label:
            if  self.tweet_id2author_label[tweet_id] in self.vocab.author_label2index:
                author_label2index[tweet_id] = self.vocab.author_label2index[self.tweet_id2author_label[tweet_id]]
            else:
                author_label2index[tweet_id] = self.vocab.len_author
        return author_label2index
    def issue_formating(self):
        issue2index = {}
        for tweet_id in self.tweet_id2issue:
            if  self.tweet_id2issue[tweet_id] in self.vocab.issue2index:
                issue2index[tweet_id] = self.vocab.issue2index[self.tweet_id2issue[tweet_id]]
            else:
                issue2index[tweet_id] = self.vocab.len_issue
        return issue2index
    def text_formating(self):
        tokenizer = TweetTokenizer()
        text2vec = {}
        if self.vocab.embed is None:
            temp = [self.vocab.len_vocab]*self.vocab.max_len
        else:
            temp = [ self.vocab.embed[-2] ] * self.vocab.max_len
        for tweet_id in self.tweet_id2text:
            word_list = tokenizer.tokenize(self.tweet_id2text[tweet_id])
            word_list = word_list[:self.vocab.max_len]
            cur = np.array(temp)
            for i,word in enumerate(word_list):
                if i>=self.vocab.max_len:
                    break
                if self.vocab.embed is None:
                    if word in self.vocab.word2index:
                        cur[i]=self.vocab.word2index[word]
                    else:
                        cur[i]=self.vocab.len_vocab+1
                else:
                    if word in self.vocab.combined_vocab.word2index:
                        cur[i]=self.vocab.embed[self.vocab.combined_vocab.word2index[word]]
                    else:
                        cur[i]=self.vocab.embed[-1]
            text2vec[tweet_id] = np.concatenate(cur)
        self.input_size = text2vec[tweet_id].shape[0]
        return text2vec
    def one_hot(self, x, num_class=18):
        output = np.zeros((num_class,1))
        output[x][0] = 1.0
        return output
    def vectorize(self):
        data = []
        for tweet_id in self.tweet_id2text:
            data.append( (tweet_id ,self.author_label2index[tweet_id] ,  self.issue2index[tweet_id], self.text2vec[tweet_id].reshape((self.input_size,1)), self.one_hot( self.label2index[tweet_id]) ) )
        return data

def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label
    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue
        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label
    #print("Read", row_count, "data points...")
    return tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):
    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])

class LossFunction():
    def __init__(self):
        pass
    def x_entropy(self, pred, target, eps = 1e-8):
        return -np.sum(np.multiply(target, np.log(pred+eps)))
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()    
    def forward(self, logits, target):
        self.target = target
        self.pred = self.softmax(logits)
        self.loss = self.x_entropy(self.pred, target)
        return self.pred, self.loss
    def backward(self):
        return np.multiply(-self.target,(1-self.pred))#-self.target+ self.pred

class Relu():
    def __init__(self, leak = 0):
        self.leak =leak
        pass
    def forward(self, x):
        self.x= x
        return np.where(x>0,x, x * self.leak)
    def backward(self, input_grad):
        return np.where(self.x>0, input_grad, input_grad* self.leak)

class Linear():
    def __init__(self, input_dim, output_dim, lr, reg, l):
        self.weights = np.random.randn(input_dim, output_dim)#np.random.normal(0.0, 1.0,(input_dim, output_dim))
        self.bias = np.random.randn(output_dim,1)#np.random.normal(0.0, 1.0,(output_dim,1))
        self.lr = lr
        self.l= l
        self.reg=reg
    # def normalize(self, x):
    #     x_norm=(np.sqrt(np.sum(x**2, 1))).repeat(x.shape[1]).reshape(x.shape)
    #     x = np.where( x_norm>1.0, x/x_norm , x )
    #     return x
    def forward(self, x):
        self.x = x
        return np.matmul(self.weights.T, x) +self.bias
    def backward(self, input_grad):
        x_grad = np.matmul(self.weights, input_grad)
        self.weight_grad = (np.matmul(self.x, input_grad.T))
        self.bias_grad = input_grad
        return x_grad
    def update(self):
        if self.reg == "l2":
            self.weight_grad += self.l * self.weights
        ### Cliping the gradients was found to be essential to solve exploding gradient issue
        # self.weights -= np.clip(self.normalize(self.weight_grad *self.lr),-0.5, 0.5)
        # self.bias -= np.clip(self.normalize(self.bias_grad * self.lr),-0.5, 0.5)
        self.weights -= np.clip(self.weight_grad *self.lr,-0.5, 0.5)
        self.bias -= np.clip(self.bias_grad * self.lr,-0.5, 0.5)
    def drop_lr(self):
        self.lr=self.lr/10
        

class Logistic_regression():
    def __init__(self, input_dim, output_dim, lr=0.01, regulariser = None, l = None):
        self.linear =  Linear(input_dim= input_dim, output_dim= output_dim, lr=0.01, reg=regulariser, l=l)
        self.criterion = LossFunction()
        self.l = l
        self.regulariser = regulariser
        self.lr=lr
    def forward(self, x, y):
        logits = self.linear.forward(x)
        pred, self.loss = self.criterion.forward(logits, y) 
        return pred
    def backward(self):
        # backpropgate to compute the gradients
        input_grad = self.criterion.backward()
        _ = self.linear.backward(input_grad)
        return
    def update(self):
        #update weights 
        self.linear.update()
        return
    def reg_loss(self):
        # add regularisation loss term to the loss
        if self.regulariser == "l2":
            self.loss += np.sum(self.linear.weights**2)*self.l
        return
    def drop_lr(self):
        self.linear.drop_lr()
        return

class dropout():
    def __init__(self, p=0.0):
        self.p = p
    def forward(self, x):
        length = x.shape[0]
        self.index = np.random.randint(length, size=int(self.p * length ))
        if len(self.index)<1:
            x[self.index] = x[self.index]*0.0
        return x
    def backward(self, input_grad):
        if len(self.index)<1:
            input_grad[self.index]= input_grad[self.index]*0.0
        return input_grad

class Neural_network():
    def __init__(self, input_dim, hidden_state, output_dim, lr=0.01, regulariser = None, l = None, p=0):
        self.linear1 =  Linear(input_dim= input_dim, output_dim= hidden_state, lr=lr, reg=regulariser, l=l)
        self.relu1 = Relu()
        self.dropout1 = dropout(p=p)
        self.linear2 =  Linear(input_dim= hidden_state, output_dim= output_dim, lr=lr*10, reg=regulariser, l=l)
        self.criterion = LossFunction()
        self.l = l
        self.regulariser = regulariser
        self.lr=lr
    def forward(self, x, y):
        h = self.dropout1.forward(self.relu1.forward(self.linear1.forward(x)))
        logits = self.linear2.forward(h)
        pred, self.loss = self.criterion.forward(logits, y) 
        return pred
    def backward(self):
        # backpropgate to compute the gradients
        input_grad = self.criterion.backward()
        input_grad = self.linear2.backward(input_grad)
        input_grad = self.dropout1.backward(input_grad)
        input_grad = self.relu1.backward(input_grad)
        _ = self.linear1.backward(input_grad)
        return
    def reg_loss(self):
        # add regularisation loss term to the loss
        if self.regulariser == "l2":
            self.loss += np.sum(self.linear1.weights**2)*self.l + np.sum(self.linear2.weights**2)*self.l
        return
    def update(self):
        #update weights 
        self.linear1.update()
        self.linear2.update()
        return
    def drop_lr(self):
        self.linear1.drop_lr()
        self.linear2.drop_lr()

class Neural_network2():
    def __init__(self, input_dim, hidden_state, output_dim, lr=0.01, regulariser = None, l = None, p=0):
        self.linear1 =  Linear(input_dim= input_dim, output_dim= hidden_state[0], lr=lr, reg=regulariser, l=l)
        self.relu1 = Relu()
        self.dropout1 = dropout(p=p)
        self.linear2 =  Linear(input_dim= hidden_state[0], output_dim= hidden_state[1], lr=lr*10, reg=regulariser, l=l)
        self.relu2 = Relu()
        self.dropout2 = dropout(p=p)
        self.linear3 =  Linear(input_dim= hidden_state[1], output_dim= hidden_state[2], lr=lr*100, reg=regulariser, l=l)
        self.relu3 = Relu()
        self.dropout3 = dropout(p=p)
        self.linear4 =  Linear(input_dim= hidden_state[2], output_dim= output_dim, lr=lr*1000, reg=regulariser, l=l)
        # self.relu4 = Relu()
        # self.dropout4 = dropout(p=p)
        # self.linear5 =  Linear(input_dim= hidden_state[3], output_dim= output_dim, lr=0.01, reg=regulariser, l=l)
        self.criterion = LossFunction()
        self.l = l
        self.regulariser = regulariser
        self.lr=lr
    def forward(self, x, y):
        x = self.dropout1.forward(self.relu1.forward(self.linear1.forward(x)))
        x = self.dropout2.forward(self.relu2.forward(self.linear2.forward(x)))
        x = self.dropout3.forward(self.relu3.forward(self.linear3.forward(x)))
        #x = self.dropout4.forward(self.relu4.forward(self.linear4.forward(x)))
        logits = self.linear4.forward(x)
        pred, self.loss = self.criterion.forward(logits, y) 
        return pred
    def backward(self):
        # backpropgate to compute the gradients
        input_grad = self.criterion.backward()
        # input_grad = self.linear5.backward(input_grad)
        # input_grad = self.dropout4.backward(input_grad)
        # input_grad = self.relu4.backward(input_grad)
        input_grad = self.linear4.backward(input_grad)
        input_grad = self.dropout3.backward(input_grad)
        input_grad = self.relu3.backward(input_grad)
        input_grad = self.linear3.backward(input_grad)
        input_grad = self.dropout2.backward(input_grad)
        input_grad = self.relu2.backward(input_grad)
        input_grad = self.linear2.backward(input_grad)
        input_grad = self.dropout1.backward(input_grad)
        input_grad = self.relu1.backward(input_grad)
        input_grad = self.linear1.backward(input_grad)
        return
    def reg_loss(self):
        # add regularisation loss term to the loss
        if self.regulariser == "l2":
            self.loss += np.sum(self.linear1.weights**2)*self.l + np.sum(self.linear2.weights**2)*self.l + np.sum(self.linear3.weights**2)*self.l+ np.sum(self.linear4.weights**2)*self.l
        return
    def update(self):
        #update weights 
        self.linear1.update()
        self.linear2.update()
        self.linear3.update()
        self.linear4.update()
        #self.linear5.update()
        return

    def drop_lr(self):
        self.linear1.drop_lr()
        self.linear2.drop_lr()
        self.linear3.drop_lr()
        self.linear4.drop_lr()
        return


class F1_score():
    def __init__(self):
        self.cm = np.zeros((17,17))       
    def add_data(self, y_pred, y_true):
        for pred, target in zip(y_pred, y_true): 
            self.cm[pred-1][target-1]+=1
    def compute(self):
        tp= np.diag(self.cm)
        precision = np.where( self.cm.sum(0)==0 ,np.zeros_like(tp), tp / self.cm.sum(0)) 
        recall = np.where(self.cm.sum(1)==0 ,np.zeros_like(tp), tp / self.cm.sum(1)) 
        f1_sore = np.where( (precision+recall)==0 , np.zeros_like(tp), 2 * np.multiply(precision , recall)/(precision+recall) )
        return f1_sore

def core(lr=0.01, regulariser = "l2", l=0.1, p=0.1, epochs=100, embed_algo = "glove", hidden_state=4096, load= True, cross_k = 1, network = "nn"):
    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')
    # Load Vocab if computed before
    if load:
        Vocab, embed_size  = pickle.load(open("./Vocab/train.vocab", "rb"))
    else:
        if embed_algo =="glove":
            combined_vocab= glove_vocab()
            embed_matrix =combined_vocab.embed_matrix
        else:
            ### This part of code deals with word2vec and is not used in the final submission
            embed_matrix, context_matrix,combined_vocab = pickle.load(open("./External_Data/representation.embedding", "rb"))
        Vocab = small_vocab (train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, embed = embed_matrix , combined_vocab = combined_vocab )
        embed_size = embed_matrix.shape[1] 
        with open("./Vocab/train.vocab", "wb") as f:   #Pickling
                    pickle.dump((Vocab, embed_size) , f)
    
    #Format the training data according to vocab
    training_loader=dataset(train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label, Vocab)
    overall_trainset = training_loader.data
    
    
    # set initial variables 
    train_total = len(overall_trainset)
    train_correct = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    if cross_k!=1:
        val_correct = np.zeros(epochs)
        val_loss = np.zeros(epochs)
    f1_var = F1_score()
    # Initialize the models for each issue and author
    model = {}
    for i in range(Vocab.len_author):
        model[i]={}
        for j in range(Vocab.len_issue):
            if network == "nn":
                model[i][j] = Neural_network(input_dim=int(Vocab.max_len* embed_size), hidden_state= hidden_state, output_dim=18, lr=lr, regulariser=regulariser, l=l, p =p)
            elif network == "nn2":
                model[i][j] = Neural_network2(input_dim=int(Vocab.max_len* embed_size), hidden_state= hidden_state, output_dim=18, lr=lr, regulariser=regulariser, l=l, p =p)
            elif network == "lr":
                model[i][j] = Logistic_regression(input_dim=int(Vocab.max_len* embed_size), output_dim=18, lr=lr, regulariser=regulariser, l=l)
            else:
                raise ValueError("Network "+network+ " not defined")
    
    # Training the classifier
    for k in range(cross_k):
        # set the train and validation set for k fold cross validation
        if cross_k!=1:
            valset = overall_trainset[k*int(train_total/cross_k):(k+1)*int(train_total/cross_k)]
            trainset = overall_trainset[:k*int(train_total/cross_k)]+overall_trainset[(k+1)*int(train_total/cross_k):]
            train_size= len(trainset)
            val_size = len(valset)
        else:
            trainset =overall_trainset
            train_size =train_total
            valset = []
            val_size=0
        # Train and validate on current trainset and validation set
        #for iterations in tqdm(range(epochs)):
        for iterations in range(epochs):
            correct=0.0
            for tweet_id, author, issue, x, y in trainset:
                pred=model[author][issue].forward(x, y)
                if np.argmax(pred) == np.argmax(y):
                    correct+=1
                model[author][issue].reg_loss()
                model[author][issue].backward()
                model[author][issue].update()
                train_loss[iterations]  += model[author][issue].loss/train_size
            train_correct[iterations] += 100 * correct/ train_size
            if cross_k!=1:
                correct=0.0
                for tweet_id, author, issue, x, y in valset:
                    pred = model[author][issue].forward(x, y)
                    if np.argmax(pred) == np.argmax(y):
                        correct+=1
                    model[author][issue].reg_loss()
                    val_loss[iterations]  += model[author][issue].loss/val_size
                val_correct[iterations] += 100 * correct/ val_size
            if iterations==99 or iterations==149:
                for i in range(Vocab.len_author):
                    for j in range(Vocab.len_issue):
                        model[i][j].drop_lr()
        if cross_k!=1:
            # add the current models output to the F1 score class to build a confusion matrix
            y_pred = []
            y_true = []
            for tweet_id, author, issue, x, y in valset:
                pred = model[author][issue].forward(x, y)
                y_pred.append(np.argmax(pred)) 
                y_true.append(np.argmax(y))
            f1_var.add_data(y_pred, y_true)
            print(val_correct[-1]/(k+1), train_correct[-1]/(k+1))
            for i in range(Vocab.len_author):
                for j in range(Vocab.len_issue):
                    if network == "nn":
                        model[i][j] = Neural_network(input_dim=int(Vocab.max_len* embed_size), hidden_state= hidden_state, output_dim=18, lr=lr, regulariser=regulariser, l=l, p =p)
                    elif network == "nn2":
                        model[i][j] = Neural_network2(input_dim=int(Vocab.max_len* embed_size), hidden_state= hidden_state, output_dim=18, lr=lr, regulariser=regulariser, l=l, p =p)
                    else:
                        model[i][j] = Logistic_regression(input_dim=int(Vocab.max_len* embed_size), output_dim=18, lr=lr, regulariser=regulariser, l=l)
                    
            
    if cross_k!=1:
        train_correct=train_correct/cross_k
        val_correct=val_correct/cross_k
        #print accuracies
        print("Train Accuracy: {} \nValidation Accuracy: {}".format(train_correct[-1],val_correct[-1]))
        # print the F1 score
        f1_score= f1_var.compute()
        print(f1_score.mean())
        print(f1_score)
        #Plot average train and validation error with epochs
        plt.plot( range(epochs), train_loss, label="Train Accuracy")
        plt.plot( range(epochs), val_loss, label="Validation Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and validation loss vs epochs at {}'.format(lr))
        plt.legend()
        if hidden_state!=4096:
            if network =="nn":
                plt.savefig("./Images/loss_"+network+"_{}_{:.6f}.png".format(hidden_state,f1_score.mean()))
            else:
                plt.savefig("./Images/loss_"+network+"_ld_{}_{:.6f}.png".format(l,f1_score.mean()))
        else:
            plt.savefig("./Images/loss_"+network+"_ld_{}_{:.6f}.png".format(l,f1_score.mean()))
        plt.close()
        #Plot average train and validation error with epochs
        plt.plot( range(epochs), train_correct, label="Train Accuracy")
        plt.plot( range(epochs), val_correct, label="Validation Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train and validation accuracy vs epochs at {}'.format(lr))
        plt.legend()
        if hidden_state!=4096:
            if network =="nn":
                plt.savefig("./Images/accuracy_"+network+"_{}_{:.6f}.png".format(hidden_state,f1_score.mean()))
            else:
                plt.savefig("./Images/accuracy_"+network+"_ld_{}_{:.6f}.png".format(l,f1_score.mean()))
        else:
            plt.savefig("./Images/accuracy_"+network+"_ld_{}_{:.6f}.png".format(l,f1_score.mean()))
        plt.close()
    else:
        # Read test data
        test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
        test_loader = dataset(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, Vocab) 
        testset = test_loader.data
        # Predict test data by learned model
        for tweet_id, author, issue, x, y in testset:
            pred=model[author][issue].forward(x, y)
            label = np.argmax(pred)
            test_tweet_id2label[tweet_id]=label
        # Save predicted labels in 'test_lr.csv'
        SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_'+network+'.csv')


def LR(lr=0.05, regulariser = "l2", l=0.1, epochs=75, embed_algo = "glove", load=True, cross_k = 1):
    core(lr=lr, regulariser = regulariser, l=l, epochs=epochs, embed_algo =embed_algo, load=load, cross_k = cross_k, network= "lr")
    return

def NN(lr=0.01, regulariser = "l2", l=0.1, p= 0.2, epochs=150, embed_algo = "glove", hidden_state=128, load= True, cross_k = 1):
    core(lr=lr, regulariser = regulariser, l=l, p= p,epochs=epochs, embed_algo = embed_algo, load=load, cross_k = cross_k, network= "nn", hidden_state=hidden_state)
    return

def NN2(lr=1e-8, regulariser = "l2", l=0.1, p= 0.2, epochs=150, embed_algo = "glove", hidden_state=[128, 64, 32], load= True, cross_k = 1):
    core(lr=lr, regulariser = regulariser, l=l, p= p,epochs=epochs, embed_algo = embed_algo, load=load, cross_k = cross_k, network= "nn2", hidden_state=hidden_state)
    return

if __name__ == '__main__':
    '''
    The following lines are used for tuning the hypreparameters.
    Uncomment this part if you need to read the vectors from ./glove/vector.txt
    '''
    import argparse
    parser = argparse.ArgumentParser( description='HW 1', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--lr',                   default="0.01",          type=str,      help='Learning Rate')
    parser.add_argument('-r', '--regulariser',           default="l2",          type=str,        help='l2, or None regulariser')
    parser.add_argument('-lm', '--lam',                  default=0.1,           type=float,      help='lambda for  regulariser')
    parser.add_argument('-e', '--epochs',                default=200,            type=int,        help='Set number of epochs')
    parser.add_argument('-ea', '--embed_algo',           default="glove",       type=str,        help='Embedding algorithm')
    parser.add_argument('-hs', '--hidden_state',         default="128", type=str,        help='number of hidden state')
    parser.add_argument('-ht', '--hidden_tuning',         default=None, type=str,        help='number of hidden state')
    parser.add_argument('-cv', '--compute_vocab',        default=False,         type=bool,       help='true if vocab is to be computed')
    parser.add_argument('-ck', '--cross_k',              default=10,             type=int,        help='k for k fold cross validation')
    parser.add_argument('-p', '--p',                     default=0.2,           type=float,      help='p for dropout ')
    parser.add_argument('-n', '--network',               default="nn",          type=str,        help='network')
    global args
    args = parser.parse_args()
    print(args)
    args.lr = [float(i) for i in args.lr.split(",")]
    args.hidden_state = [int(h.strip()) for h in args.hidden_state.split(",")]
    if args.hidden_tuning is not None:
        epoch = {512:50, 256:100, 128:200, 64:400, 32:800}
        args.hidden_tuning = [int(h.strip()) for h in args.hidden_tuning.split(",")]
        print(args.hidden_tuning)
        for h in args.hidden_tuning:
            print("\n\n\nRunning Neural Network: hidden neurons {}".format(h))
            NN(lr=args.lr[-1], regulariser = args.regulariser, l=args.lam, p= args.p, epochs=epoch[h], embed_algo = args.embed_algo,load = not args.compute_vocab, hidden_state= h, cross_k= args.cross_k)           
    elif len(args.lr)>1:
        print(args.lr)
        for lr_cur in args.lr:
            if args.network == "lr":
                print("\n\n\nRunning Logistic Regression: lambda {}".format(lr_cur))
                LR(lr=0.5, regulariser = args.regulariser, l=lr_cur, epochs=args.epochs, embed_algo = args.embed_algo, load = not args.compute_vocab, cross_k= args.cross_k)
            else:
                print("\n\n\nRunning Neural Network 2:  Lambda {}".format(lr_cur))
                NN2(lr=lr_cur, regulariser = args.regulariser, l=args.lam, p= args.p,epochs=args.epochs, embed_algo = args.embed_algo,load = not args.compute_vocab, hidden_state= args.hidden_state[1:], cross_k= args.cross_k)
    else:
        lr_cur = args.lr[0]
        if args.network =="lr":
            print("Running Logistic Regression:")
            LR(lr=lr_cur, regulariser = args.regulariser, l=args.lam, epochs=args.epochs, embed_algo = args.embed_algo, load = not args.compute_vocab, cross_k= args.cross_k)
        elif args.network =="nn":
            print("Running Neural Network:")
            NN(lr=lr_cur, regulariser = args.regulariser, l=args.lam, p= args.p, epochs=args.epochs, embed_algo = args.embed_algo,load = not args.compute_vocab, hidden_state= args.hidden_state[0], cross_k= args.cross_k)
        elif args.network =="nn2":
            print("Running Neural Network 2:")
            NN2(lr=10e-8, regulariser = args.regulariser, l=args.lam, p= args.p,epochs=args.epochs, embed_algo = args.embed_algo,load = not args.compute_vocab, hidden_state= args.hidden_state[1:], cross_k= args.cross_k)
        else:
            LR()
            NN()
        
    # # '''
    # # Final Run
    # # '''
    # print("Running Logistic Regression:")
    # LR()
    # print("Running Neural Network:")
    # NN()
