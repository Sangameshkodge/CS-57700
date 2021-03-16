import json
import pickle
from nltk.tokenize import TweetTokenizer
dataset = []
with open("./External_Data/tweets.json") as f:
    line = f.readline()
    while line:
        cur_data = json.loads(line)
        cur_data = cur_data["text"].split(" ")
        line = f.readline()
        # while len(cur_data) > train_vocab.max_len:
        #     dataset.append(" ".join(cur_data[:train_vocab.max_len]))
        #     cur_data = cur_data[train_vocab.max_len:]
        dataset.append(" ".join(cur_data))
with open("./External_Data/corpus.txt", "w") as f:
    f.writelines("%s\n" % line for line in dataset)
with open("./glove/corpus.txt", "w") as f:
    f.writelines("%s\n" % line for line in dataset)

class big_vocab():
    def __init__(self, data, vocab_size=None, min_freq=5):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.tokenizer = TweetTokenizer()
        self.word2index, self.index2word, self.index2freq, self.max_len = self.build(
            data)

    def build(self, data):
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
                    # for word in train_vocab.word2index:
                    #     if word not in word2index and word in word_freq:
                    #         index += 1
                    #         word2index[word] = index
                    #         index2word[index] = word
                    #         index2freq[index] = freq
                    self.vocab_size = index
                    return word2index, index2word, index2freq, max_len
            if self.min_freq is not None:
                if freq < self.min_freq:
                    # for word in train_vocab.word2index:
                    #     if word not in word2index and word in word_freq:
                    #         index += 1
                    #         word2index[word] = index
                    #         index2word[index] = word
                    #         index2freq[index] = freq
                    self.vocab_size = index
                    return word2index, index2word, index2freq, max_len
            word2index[word] = index
            index2word[index] = word
            index2freq[index] = freq
        self.vocab_size = len(ordered_vocab)
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

vocab = big_vocab(dataset, vocab_size=None, min_freq=5)
with open("./Vocab/tweets.vocab", "wb") as f:  # Pickling
    pickle.dump(vocab, f)

# ordered_word_freq = pickle.load(open("./Vocab/ordered_word_freq.vocab", "rb"))
# with open("./glove/vocab.txt", "w") as f: 
#     f.writelines(word +" "+ str(freq)+"\n"  for word, freq in ordered_word_freq) 

    
