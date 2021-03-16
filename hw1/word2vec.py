from nltk.tokenize import TweetTokenizer
import pickle
import numpy as np
from tqdm import tqdm
import copy
import random


class big_vocab():
    def __init__(self, data, vocab_size=None, min_freq=None, train_vocab=None):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.tokenizer = TweetTokenizer()
        self.word2index, self.index2word, self.index2freq, self.max_len = self.build(
            data, train_vocab)

    def build(self, data, train_vocab):
        word_freq = {}
        max_len = 0
        for tweet in data:
            cur_tweet = self.tokenizer.tokenize(tweet)
            if max_len < len(cur_tweet):
                max_len = len(cur_tweet)
            for token in cur_tweet:
                if token in word_freq:
                    word_freq[token] += 1
                else:
                    word_freq[token] = 1

        ordered_vocab = sorted(
            word_freq.items(), key=lambda t: t[1], reverse=True)
        with open("./Vocab/ordered_word_freq.vocab", "wb") as f:  # Pickling
            pickle.dump(ordered_vocab, f)

        word2index = {}
        index2word = {}
        index2freq = {}
        for index, (word, freq) in enumerate(ordered_vocab):
            if self.vocab_size is not None:
                if self.vocab_size <= index:
                    for word in train_vocab.word2index:
                        if word not in word2index and word in word_freq:
                            index += 1
                            word2index[word] = index
                            index2word[index] = word
                            index2freq[index] = freq
                    self.vocab_size = index
                    return word2index, index2word, index2freq, max_len
            if self.min_freq is not None:
                if freq < self.min_freq:
                    for word in train_vocab.word2index:
                        if word not in word2index and word in word_freq:
                            index += 1
                            word2index[word] = index
                            index2word[index] = word
                            index2freq[index] = freq
                    self.vocab_size = index
                    return word2index, index2word, index2freq, max_len
            word2index[word] = index
            index2word[index] = word
            index2freq[index] = freq
        self.vocab_size = index
        return word2index, index2word, index2freq, max_len

    def get_index_from_tweet(self, tweet, padding=False):
        cur_tweet = self.tokenizer.tokenize(tweet)
        if padding:
            output = [self.vocab_size] * self.max_len
            if len(cur_tweet) > self.max_len:
                cur_tweet = cur_tweet[:self.max_len]
        else:
            output = [self.vocab_size] * len(cur_tweet)
        for i, token in enumerate(cur_tweet):
            if token in self.word2index:
                output[i] = self.word2index[token]
            else:
                # Out of vocab token
                output[i] = self.vocab_size+1
        return output

    def get_index_from_data(self, data, padding=False, convert_data=False):
        if convert_data:
            output = []
            for cur_data in data:
                output.append(self.get_index_from_tweet(cur_data))
            with open("./External_Data/indextweets.data", "wb") as f:  # Pickling
                pickle.dump(output, f)
        else:
            output = pickle.load(
                open("./External_Data/indextweets.data", "rb"))
        return output


class representation_learner():
    def __init__(self, data, vocab, embed_len=768, epoch=10, lr=0.01, convert_data=True, load_embed=True):
        self.embed_len = embed_len
        if load_embed:
            self.embed_matrix, self.context_matrix, self.vocab = pickle.load(
                open("./External_Data/representation.embedding", "rb"))
        else:
            self.embed_matrix = np.random.rand(vocab.vocab_size+2, embed_len)
            self.context_matrix = np.random.rand(vocab.vocab_size+2, embed_len)
            self.vocab = vocab
        self.epoch = epoch
        self.lr = lr
        data = self.vocab.get_index_from_data(
            data=data, convert_data=convert_data)
        temp = list(self.vocab.index2freq.values())
        temp.append(temp[self.vocab.vocab_size-1])
        temp.append(temp[self.vocab.vocab_size-1])
        self.propbability = np.array(temp)
        self.train(data=data)

    def preprocess(self, data):
        dataset = []
        # max_len= 0
        for tweet in tqdm(data):
            if len(tweet) < 2:
                continue
            # if max_len<len(tweet):
            #     max_len = len(tweet)
            for index, word in enumerate(tweet):
                # Make a array of positive and negative samples
                positive_samples = np.delete(tweet, [index])
                # Make soft labels
                cur_probab = [
                    1/(2**abs(val)) for val in range(-index, len(tweet)-index, 1) if val != 0]
                cur_probab = self.softmax(cur_probab)
                target = np.zeros((self.vocab.vocab_size+2,))
                target[positive_samples] = cur_probab
                dataset.append((index, tweet, target))
        # print(max_len)
        return dataset

    def normalize(self, x):
        return x / np.sqrt(np.sum(x**2, 1)).repeat(self.embed_len).reshape(x.shape)

    def train(self, data):
        for iter in range(self.epoch):
            running_loss = 0
            counter = 0
            sampled_data = random.sample(data, 10000)
            processed_sampled_data = self.preprocess(sampled_data)
            for index, tweet, target in tqdm(processed_sampled_data):
                word = tweet[index]
                positive_samples = np.delete(tweet, [index])
                negative_samples = np.random.choice(np.delete(np.arange(self.vocab.vocab_size+2), tweet), size=4*len(
                    positive_samples), replace=False, p=self.norm(np.delete(self.propbability, tweet)))

                # forward pass through the autoencoder
                self.target = target
                self.pred = self.forward(word)

                # Compute the loss
                running_loss += self.loss_function()
                counter += 1

                # Update the weights using backprop
                self.backward(positive_samples, negative_samples, word)
                self.context_matrix = self.normalize(self.context_matrix)
                self.embed_matrix = self.normalize(self.embed_matrix)
            with open("./External_Data/representation.embedding", "wb") as f:  # Pickling
                pickle.dump(
                    (self.embed_matrix, self.context_matrix, self.vocab), f)
            print("Loss at epoch {} is {}".format(
                iter, float(running_loss) / float(counter)))
        return

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def norm(self, x):
        return x/x.sum()

    def forward(self, word):
        self.hidden_state = self.embed_matrix[word]
        self.logits = np.matmul(self.context_matrix, self.hidden_state)
        pred = self.softmax(self.logits)
        return pred

    def loss_function(self, eps=1e-8):
        self.loss = -np.sum(np.multiply(self.target, np.log(self.pred+eps)) +
                            np.multiply(1-self.target, np.log(1-self.pred + eps)))
        return self.loss

    def backward(self, positive_samples, negative_samples, word):
        # set gradients to 0
        self.grad_embed_matrix = np.zeros_like(self.embed_matrix)
        self.grad_context_matrix = np.zeros_like(self.context_matrix)
        self.grad_hidden_state = np.zeros_like(self.hidden_state)

        # Compute gradients
        self.grad_logits = -self.target + self.pred
        for pos_sample in positive_samples:
            self.grad_context_matrix[pos_sample] = self.hidden_state * \
                self.grad_logits[pos_sample]
            self.grad_hidden_state += self.context_matrix[pos_sample] * \
                self.grad_logits[pos_sample]
        for neg_sample in negative_samples:
            self.grad_context_matrix[neg_sample] = self.hidden_state * \
                self.grad_logits[neg_sample]
            self.grad_hidden_state += self.context_matrix[neg_sample] * \
                self.grad_logits[neg_sample]
        self.grad_embed_matrix[word] = self.grad_hidden_state
        # update weights
        self.embed_matrix -= self.grad_embed_matrix * self.lr
        self.context_matrix -= self.grad_context_matrix * self.lr
        return


if __name__ == "__main__":
    import json

    import argparse
    parser = argparse.ArgumentParser(
        description='Word2Vec', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs',                default=1000,
                        type=int,        help='Set number of epochs')
    parser.add_argument('-l', '--lr',                    default=0.01,
                        type=float,      help='Learning Rate')
    parser.add_argument('-vs', '--vocab_size',
                        default=15000,       type=int,        help='Vocab_size')
    parser.add_argument('-m', '--min_freq',              default=100,
                        type=int,        help='minimum freq')
    parser.add_argument('-hs', '--hidden_size',          default=256,
                        type=int,        help='hidden dimension')
    parser.add_argument('-bv', '--build_vocab',          default=False,
                        type=bool,       help='load vocab from file')
    parser.add_argument('-bd', '--build_dataset',        default=False,
                        type=bool,       help='load dataset from file')
    parser.add_argument('-bid', '--build_ind_data',      default=False,
                        type=bool,       help='load index converted from file')
    #parser.add_argument('-bpd','--build_processed_data',      default=False,       type=bool,       help='load processed dataset from file')
    parser.add_argument('-le', '--load_embeddings',      default=False,
                        type=bool,       help='load ckpt from file')
    global args

    args = parser.parse_args()
    print(args)
    if args.build_dataset or args.build_vocab:
        from main import small_vocab
        train_vocab = pickle.load(open("./Vocab/train.vocab", "rb"))

    if args.build_dataset:
        dataset = []
        with open("./External_Data/tweets.json") as f:
            line = f.readline()
            while line:
                cur_data = json.loads(line)
                cur_data = cur_data["text"].split(" ")
                line = f.readline()
                while len(cur_data) > train_vocab.max_len:
                    dataset.append(" ".join(cur_data[:train_vocab.max_len]))
                    cur_data = cur_data[train_vocab.max_len:]
                dataset.append(" ".join(cur_data) )

        with open("./External_Data/text.data", "wb") as f:  # Pickling
            pickle.dump(dataset, f)
        with open("./External_Data/corpus.txt", "w") as f:
           f.writelines("%s\n" % line for line in dataset)
    else:
        dataset = pickle.load(open("./External_Data/text.data", "rb"))
        
            


    if args.build_vocab:
        combined_vocab = big_vocab(
            dataset, vocab_size=args.vocab_size, min_freq=args.min_freq,  train_vocab=train_vocab)
        with open("./Vocab/tweets.vocab", "wb") as f:  # Pickling
            pickle.dump(combined_vocab, f)
    else:
        combined_vocab = pickle.load(open("./Vocab/tweets.vocab", "rb"))

    Embedding_learner = representation_learner(data=dataset, vocab=combined_vocab, embed_len=args.hidden_size,
                                               epoch=args.epochs, lr=args.lr, convert_data=args.build_ind_data, load_embed=args.load_embeddings)
