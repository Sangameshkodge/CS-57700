import numpy as np
import string
import argparse
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import os

import matplotlib.pyplot as plt
  
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


'''Following are some helper functions from https://github.com/lixin4ever/E2E-TBSA/blob/master/utils.py to help parse the Targeted Sentiment Twitter dataset. You are free to delete this code and parse the data yourself, if you wish.

You may also use other parsing functions, but ONLY for parsing and ONLY from that file.
'''
def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')
                if tag == 'O':
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset


def train_preprocess(dataset, embedding, wv_from_bin, raw_tag_2_label, option, device):
    ###Generating sequential dataset for training
    train_loader = []
    for idx,data in enumerate(dataset):
        x = torch.tensor([], device = device).long()
        y = torch.tensor([], device = device).long()
        name_list = {}
        for word, tag in zip(data["words"], data["ts_raw_tags"]):
            y = torch.cat((y, torch.tensor(raw_tag_2_label[tag], device= device).unsqueeze(0)), dim=0)
            if option==2:
                # if raw_tag_2_label[tag] == 0:
                if word not in embedding.word2idx.keys():
                    try:
                        current = torch.tensor(wv_from_bin[word], device= device).unsqueeze(0)
                        embedding.add_word(word)
                        embedding.embedding.weight[ embedding.word2idx[word] ] = current#(current-current.mean())/current.std()
                        set_x = True
                        x = torch.cat((x, embedding.word2idx[word].unsqueeze(0)), dim=0)
                    except:
                        set_x = False
                else:
                    x = torch.cat((x, embedding.word2idx[word].unsqueeze(0)), dim=0)
                    set_x = True
                # else:
                #     set_x=False
            else:   
                if raw_tag_2_label[tag] == 0:
                    if word not in embedding.word2idx.keys():
                        embedding.add_word(word)
                    x = torch.cat((x, embedding.word2idx[word].unsqueeze(0)), dim=0)
                    set_x = True
                else:
                    set_x = False

            if not set_x:
                if word in name_list:
                    x = torch.cat((x, name_list[word].unsqueeze(0)), dim=0)
                else:
                    if len(name_list) >= embedding.unk_words:
                        x = torch.cat((x, torch.tensor(embedding.unk_words, device = device).unsqueeze(0)), dim=0)
                    else:
                        name_list[word] = torch.tensor(len(name_list), device = device)
                        x = torch.cat((x, name_list[word].unsqueeze(0)), dim=0)
        
        y_prev = torch.cat(( (-1)*torch.ones(1, device=device).long(),y[:-1]), dim=-1 )+1
        train_loader.append((x.long().unsqueeze(0).to(device),y.long().unsqueeze(0).to(device), y_prev.long().unsqueeze(0).to(device), idx))
    random.shuffle(train_loader)
    ### splitting the data into training and validation
    
    ### Unsequential training data for faster batched training
    separated_train_loader = {
        0:[],
        1:[],
        2:[],
        3:[],
    }
    for batch_idx, (data, tag, y_prev, idx) in enumerate(train_loader):
        data= data.squeeze(0)
        tag = tag.squeeze(0)
        y_prev = y_prev.squeeze(0)
        for word_idx in range(tag.shape[0]):
            separated_train_loader[tag[word_idx].item()].append((data[word_idx], y_prev[word_idx]))
    
    for class_num in separated_train_loader:
        random.shuffle(separated_train_loader[class_num])
    
    return train_loader,  separated_train_loader

def test_preprocess(dataset, embedding, wv_from_bin, raw_tag_2_label, option, device):
    ###Generating sequential the dataset for testing
    test_loader = []
    for idx,data in enumerate(dataset):
        x = torch.tensor([], device = device).long()
        y = torch.tensor([], device = device).long()
        name_list = {}
        for word, tag in zip(data["words"], data["ts_raw_tags"]):
            y = torch.cat((y, torch.tensor(raw_tag_2_label[tag], device= device).unsqueeze(0)), dim=0)
            
            if word in embedding.word2idx.keys():
                x = torch.cat((x, embedding.word2idx[word].unsqueeze(0)), dim=0)
                set_x = True
            else:
                if word in name_list:
                    x = torch.cat((x, name_list[word].unsqueeze(0)), dim=0)
                else:
                    if len(name_list) >= embedding.unk_words:
                        x = torch.cat((x, torch.tensor(embedding.unk_words, device = device).unsqueeze(0)), dim=0)
                    else:
                        name_list[word] = torch.tensor(len(name_list), device = device)
                        x = torch.cat((x, name_list[word].unsqueeze(0)), dim=0)
        
        y_prev = torch.cat(( (-1)*torch.ones(1, device=device).long(),y[:-1]), dim=-1 )+1
        test_loader.append((x.long().unsqueeze(0).to(device),y.unsqueeze(0).long().to(device),y_prev.unsqueeze(0).long().to(device), idx))
    return test_loader

def SaveFile(given_df, output_csv_file):

    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(["sentence", "words", "ts_raw_tags", "ts_pred_tags"])
        for index, row in given_df.iterrows():
            writer.writerow([row['sentence'], row['words'], row['ts_raw_tags'], row['ts_pred_tags']])


class TargetedSentiment(nn.Module):
    def __init__(self, num_classes = 4, batch_size = 1,  embed_dim = 300, y_embed_dim = 50, p=0.2, device = "cpu", imbalance=None, suffix=""):

        super(TargetedSentiment, self).__init__()
        self.suffix = suffix    
        self.num_classes= num_classes
        self.test_batch_size = batch_size
        self.p = p
        self.device = device
        self.append_tensor = torch.arange(0,self.num_classes,1 , device=device).expand(1,self.test_batch_size,self.num_classes).permute(1,0,2).contiguous()
        self.embed_dim = embed_dim
        self.y_embed_dim = y_embed_dim
        self.reset()
        self.scale_func(imbalance)
        

        if "small" in self.suffix :
            self.transition_emission_probabilities_estimator = nn.Linear(self.embed_dim+self.y_embed_dim, self.num_classes) 
        elif "mid" in self.suffix :
            self.Linear1 =nn.Linear(self.embed_dim+self.y_embed_dim, 256) 
            self.bn1 =torch.nn.BatchNorm1d(256)
            self.Linear2 =nn.Linear(256, 128)
            self.shortcut1= nn.Linear(self.embed_dim+self.y_embed_dim, 128) 
            self.bn2 =torch.nn.BatchNorm1d(128)
            self.transition_emission_probabilities_estimator = nn.Linear(128, self.num_classes) 
        elif "big" in self.suffix :
            self.Linear1 =nn.Linear(self.embed_dim+self.y_embed_dim, 512) 
            self.bn1 =torch.nn.BatchNorm1d(512)
            self.Linear2 =nn.Linear(512, 256)
            self.shortcut1= nn.Linear(self.embed_dim+self.y_embed_dim, 256) 
            self.bn2 =torch.nn.BatchNorm1d(256)
            self.Linear3 =nn.Linear(256, 128) 
            self.bn3 =torch.nn.BatchNorm1d(128)
            self.Linear4 =nn.Linear(128, 64) 
            self.shortcut2= nn.Linear(256, 64)
            self.bn4 =torch.nn.BatchNorm1d(64)
            self.Linear5 =nn.Linear(64, 32) 
            self.bn5 =torch.nn.BatchNorm1d(32)
            self.Linear6 =nn.Linear(32, 16) 
            self.shortcut3= nn.Linear(64, 16)
            self.bn6 =torch.nn.BatchNorm1d(16)
            self.transition_emission_probabilities_estimator = nn.Linear(16, self.num_classes)
        else:
            raise ValueError
    def scale_func(self, imbalance=None):
        if imbalance is not None:
            scale = []
            for n_samples in imbalance:
                scale.append(sum(imbalance)/(2*n_samples))
            self.scale = torch.tensor(scale, device = self.device)
        else:
            self.scale = torch.tensor([1]*self.num_classes, device = self.device)
            
    def reset(self):
        self.k=1
        self.prev_best_path = self.append_tensor#-1* torch.ones((1, self.test_batch_size, self.num_classes), device= self.device)
        self.prev_best_prob = torch.ones((self.test_batch_size,self.num_classes), device= self.device)

    def viterbi(self, StateTransitionEmissions):
        """
        StateTransitionEmissions (Tensor): batch_size x num_classes x num_classes
        prev_best_prob: batch_size x num_classes
        prev_best_path: batch_size x k x num_classes
        """
        """
        At each step the best path for the current element being each class is stored in prev_best_path 
        hence it has the dimension batch_size x k x num_classes
        the probability of each 
        This code store the best path for the current element belonging to each of the classes. 
        Hence The DP table is not required as we dont need to go trought the table for finding the best path.
        We just need to look at the final probabilities of each of the 4 classes.
        """
        ###multiply with previous probabilities
        StateTransitionEmissions= StateTransitionEmissions*self.prev_best_prob.expand(4,4).permute(1,0)

        #max prob in current path - each class considering
        max_prob_for_current_transitions = StateTransitionEmissions.max(dim=-2)
        
        #expand max prob tensor for gathering the best path for current input being considered for each class
        expanded_max_prob_index_for_current_transitions = max_prob_for_current_transitions[1].expand(self.k,self.test_batch_size,self.num_classes)
        #Gather best tensor for current element being considered as each class.
        gathered_tensor_for_best_path = torch.gather(self.prev_best_path,-1, expanded_max_prob_index_for_current_transitions)
        #append the tensor with the current class
        self.prev_best_path = torch.cat((gathered_tensor_for_best_path, self.append_tensor ), 0)
        
        # #Compute the probability for the best path without current element(gather from previous probabilites)
        # previous_path_probability = torch.gather(self.prev_best_prob,-1, max_prob_for_current_transitions[1])
        #Multiply current probabilities with current probabilities to get the probabilitiy score for each path.
        self.prev_best_prob = max_prob_for_current_transitions[0] 
        
        # update the positon processing variable
        self.k += 1
        return 

    def forward(self,  input, y_embedding =None):
        """
        input (tenor): batch_size x len_sentence x embedding_size
        y (tenor) : batch_size x len_sentence
        transitions_emission_probabilities (tenor) :  batch_size x len_sentence x num_classes
        """
        len_input = input.shape[1] 
        if self.training :
            input=input.squeeze(0)
            if "small" not in self.suffix:
                x = F.dropout(self.bn1(F.relu(self.Linear1(input))), p=self.p )
                input = F.dropout(self.bn2(F.relu(self.Linear2(x) + self.shortcut1(input))), p=self.p )
                if "mid" not in self.suffix:
                    x = F.dropout(self.bn3(F.relu(self.Linear3(input))), p=self.p )
                    input = F.dropout(self.bn4(F.relu(self.Linear4(x)+ self.shortcut2(input))), p=self.p )
                    x = F.dropout(self.bn5(F.relu(self.Linear5(input))), p=self.p )
                    input = F.dropout(self.bn6(F.relu(self.Linear6(x)+ self.shortcut3(input))), p=self.p )
            transitions_emission_probabilities = self.transition_emission_probabilities_estimator(input).unsqueeze(0)
            return transitions_emission_probabilities 
            
        else:
            self.reset()
            current_input = input[:, self.k, :].expand((1, self.test_batch_size, self.embed_dim)).permute(1,0,2).contiguous()
            y_prev = y_embedding(torch.arange(0,1,1).unsqueeze(0))
            x_y_prev_input = torch.cat((current_input, y_prev), dim=-1 ).squeeze(0)
            if "small" not in self.suffix:
                x_y_prev_x = self.bn1(F.relu(self.Linear1(x_y_prev_input)))
                x_y_prev_input = self.bn2(F.relu(self.Linear2(x_y_prev_x) + self.shortcut1(x_y_prev_input)))
                if "mid" not in self.suffix:
                    x_y_prev_x = self.bn3(F.relu(self.Linear3(x_y_prev_input)))
                    x_y_prev_input = self.bn4(F.relu(self.Linear4(x_y_prev_x)+ self.shortcut2(x_y_prev_input)))
                    x_y_prev_x = self.bn5(F.relu(self.Linear5(x_y_prev_input)))
                    x_y_prev_input = self.bn6(F.relu(self.Linear6(x_y_prev_x)+ self.shortcut3(x_y_prev_input)))
            self.prev_best_prob =  F.softmax(self.transition_emission_probabilities_estimator(x_y_prev_input).detach(), dim=-1)* self.scale

            while self.k < len_input:
                current_input = input[:, self.k, :].expand((self.num_classes, self.test_batch_size, self.embed_dim)).permute(1,0,2).contiguous()
                y_prev = y_embedding(torch.arange(1,5,1).unsqueeze(0))
                x_y_prev_input = torch.cat((current_input, y_prev), dim=-1 ).squeeze(0)
                if "small" not in self.suffix:
                    x_y_prev_x = self.bn1(F.relu(self.Linear1(x_y_prev_input)))
                    x_y_prev_input = self.bn2(F.relu(self.Linear2(x_y_prev_x) + self.shortcut1(x_y_prev_input)))
                    if "mid" not in self.suffix:
                        x_y_prev_x = self.bn3(F.relu(self.Linear3(x_y_prev_input)))
                        x_y_prev_input = self.bn4(F.relu(self.Linear4(x_y_prev_x)+ self.shortcut2(x_y_prev_input)))
                        x_y_prev_x = self.bn5(F.relu(self.Linear5(x_y_prev_input)))
                        x_y_prev_input = self.bn6(F.relu(self.Linear6(x_y_prev_x)+ self.shortcut3(x_y_prev_input)))
                transitions_emission_probabilities = F.softmax(self.transition_emission_probabilities_estimator(x_y_prev_input).detach(), dim=-1).unsqueeze(0)
                self.viterbi(transitions_emission_probabilities * self.scale)
            return self.prev_best_path[:,:,torch.max(self.prev_best_prob, dim=-1 )[1].item()].permute(1,0)

class BiLSTM(nn.Module):
    def __init__(self, embed_dim=300, hidden_dim=150, lstm_layer=1, p=0.2, device = "cpu", suffix = None):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=p)
        if suffix is not None:
            if "big" in suffix:
                lstm_layer = 6
            elif "mid" in suffix:
                lstm_layer = 2
            elif "small" in suffix:
                lstm_layer = 1

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            dropout = p,
                            bidirectional=True).to(device)
        self.device = device
    
    def forward(self, input):
        x = torch.transpose(input, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(torch.transpose(lstm_out, dim0=1, dim1=0)) 
        return out


class Embedding(nn.Module):
    def __init__(self, embed_dim = 300, max_vocab_size = 10000, unk_words = 10, device="cpu"):
        super(Embedding,self).__init__()
        self.embed_dim = embed_dim
        self.max_vocab_size = max_vocab_size
        self.embedding = nn.Embedding(self.max_vocab_size,self.embed_dim).to(device)
        self.unk_words = unk_words
        self.vocab_size = 0
        self.device =device

        self.word2idx = {}
        self.idx2word = {}
        for i in range(unk_words+1):
            self.word2idx["unk{}".format(self.vocab_size)]=torch.tensor(self.vocab_size, device= device)
            self.idx2word[self.vocab_size] = "unk{}".format(self.vocab_size)
            self.vocab_size+=1

    def add_word(self, word):
        self.word2idx[word] = torch.tensor(self.vocab_size, device= self.device)
        self.idx2word[self.vocab_size ]= word
        self.vocab_size+=1
    
    def forward(self, x):
        return self.embedding.weight[x]
    
    def normalize(self):
        pass
        #self.embedding.weight = ( (self.embedding.weight-self.embedding.weight.mean(-1).expand(self.embed_dim, self.max_vocab_size).permute(1,0))/self.embedding.weight.mean(-1).expand(self.embed_dim, self.max_vocab_size).permute(1,0) ).detach()

class metric_computations():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes))

    def cm_compute(self, output, tags):
        for i,j in zip(output.view(-1).cpu().numpy(), tags.view(-1).cpu().numpy()):
            self.cm[i,j] +=1
        #print(self.get_current_scores(output, tags))
        return

    def get_current_scores(self, output, tags):
        cm = torch.zeros((self.num_classes, self.num_classes))
        for i,j in zip(output.view(-1).cpu().numpy(), tags.view(-1).cpu().numpy()):
            cm[i.item(),j.item()] +=1
        TP = cm.diag()[1:].sum()
        FP = cm[:,1:].sum()-TP
        FN = cm[1:,:].sum()-TP
        precision = torch.where((TP+FP)!=0,TP/(TP+FP),torch.zeros_like(TP))
        recall = torch.where((TP+FN)!=0,TP/(TP+FN),torch.zeros_like(TP))
        f1 =  torch.where(precision+recall!=0,2*precision*recall/(precision+recall),torch.zeros_like(TP))
        return {
            'TP' : TP,
            'FP' : FP,
            'FN' : FN,
            "precision": precision,
            "recall" : recall,
            "f1" : f1,
        }

    def get_scores(self):
        # self.get_current_scores(torch.tensor([1,0,0,0,0,0,0]), torch.tensor([1,1,0,0,0,0,0]))
        # self.get_current_scores(torch.tensor([3,0,0]), torch.tensor([3,3,3]))
        # self.get_current_scores(torch.tensor([3,2,0]), torch.tensor([3,3,3]))
        # self.get_current_scores(torch.tensor([3,2,0]), torch.tensor([3,3,0]))
        # self.get_current_scores(torch.tensor([3,2,0]), torch.tensor([3,3,2]))
        # self.get_current_scores(torch.tensor([3,2,0]), torch.tensor([3,0,0]))
        TP = self.cm.diag()[1:].sum()
        FN = self.cm[:,1:].sum()-TP
        FP = self.cm[1:,:].sum()-TP
        precision = torch.where((TP+FP)!=0,TP/(TP+FP),torch.zeros_like(TP))
        recall = torch.where((TP+FN)!=0,TP/(TP+FN),torch.zeros_like(TP))
        f1 =  torch.where(precision+recall!=0,2*precision*recall/(precision+recall),torch.zeros_like(TP))
        acc = self.cm.diag().sum()/ self.cm.sum()
        return {
            "precision": precision,
            "recall" : recall,
            "f1" : f1,
            "acc" : acc,
        }

def train(model,
        bilstm, 
        embedding, 
        y_embedding, 
        dataset,
        valset, 
        testset,
        epochs, 
        criterion, 
        optimizer, 
        scheduler, 
        batch_size, 
        metric, 
        device, 
        option, 
        train_embedding,
        suffix, 
        file = "./models"):

    best_validation_score= 0
    training_loss_mem = []
    validation_loss_mem = []
    test_loss_mem = []
    print("Iteration: {}".format(0))
    train_loss, train_score = test(model, bilstm, embedding, y_embedding, dataset, metric, option)
    print("Train_Loss: {}, Precision: {}, Recall: {}, F1: {} Acc:{}".format(train_loss,train_score["precision"],train_score["recall"],train_score["f1"],train_score["acc"]))
    val_loss, val_score = test(model, bilstm, embedding, y_embedding, valset, metric, option)
    print("Validation_Loss: {}, Precision: {}, Recall: {}, F1: {} Acc:{}".format(val_loss,val_score["precision"],val_score["recall"],val_score["f1"],val_score["acc"]))
    test_loss, test_score = test(model, bilstm, embedding, y_embedding, testset, metric, option)
    print("Test_Loss: {}, Precision: {}, Recall: {}, F1: {} Acc:{}\n".format(test_loss,test_score["precision"],test_score["recall"],test_score["f1"],test_score["acc"]))
    training_loss_mem.append(train_loss)
    validation_loss_mem.append(val_loss)
    test_loss_mem.append(test_loss)

    for iterations in range(epochs):
        model.train()
        if bilstm!=None:
            bilstm.train()
        if option!=2 or train_embedding:
            embedding.train()
        else:
            embedding.eval()
        y_embedding.train()
        for batch_idx, (data, tag, y_prev, idx) in enumerate(dataset):
            ### zero out past gradients
            optimizer.zero_grad()
            ### zero out past gradients
            optimizer.zero_grad()
            y_prev = torch.cat(( (-1)*torch.ones((model.test_batch_size,1), device=device).long(),tag[:,:-1]), dim=-1 )+1
            y_prev = y_embedding(y_prev)
            if option ==2 and not train_embedding:
                x = embedding(data).detach()
            else:
                x = embedding(data)
            ### Forward pass
            if option == 3: 
                x = bilstm(x)
            x_y_prev = torch.cat((x, y_prev) , dim=-1) 
            output = model(x_y_prev).view(-1, model.num_classes)
            ### Loss computation
            loss = criterion(output, tag.squeeze(0))
            ### backward pass
            loss.backward()
            ###network update
            optimizer.step()

        print("Iteration: {}".format(iterations+1))
        train_loss, train_score = test(model, bilstm, embedding, y_embedding, dataset, metric, option)
        print("Train_Loss: {}, Precision: {}, Recall: {}, F1: {} Acc:{}".format(train_loss,train_score["precision"],train_score["recall"],train_score["f1"],train_score["acc"]))
        val_loss, val_score = test(model, bilstm, embedding, y_embedding, valset, metric, option)
        print("Validation_Loss: {}, Precision: {}, Recall: {}, F1: {} Acc:{}".format(val_loss,val_score["precision"],val_score["recall"],val_score["f1"],val_score["acc"]))
        test_loss, test_score = test(model, bilstm, embedding, y_embedding, testset, metric, option)
        print("Test_Loss: {}, Precision: {}, Recall: {}, F1: {} Acc:{}\n".format(test_loss,test_score["precision"],test_score["recall"],test_score["f1"],test_score["acc"]))
        training_loss_mem.append(train_loss)
        validation_loss_mem.append(val_loss)
        test_loss_mem.append(val_loss)
        scheduler.step(val_score["f1"] /val_loss)
        
        if val_score["f1"] /val_loss > best_validation_score:
            best_validation_score = val_score["f1"] /val_loss
            print("Saving model with score {}\n".format(best_validation_score))
            save_model(model, bilstm, embedding,y_embedding, option, suffix, train_embedding, file)

    return model, bilstm, embedding, y_embedding, optimizer, scheduler, training_loss_mem, validation_loss_mem, test_loss_mem

def test(model, bilstm, embedding, y_embedding, dataset, metric, option):
    model.eval()
    if bilstm!=None:
        bilstm.eval()
    embedding.eval()
    y_embedding.eval()

    loss = 0.0 
    metric.reset()    
    for batch_idx, (data, tag, y_prev, idx) in enumerate(dataset):
        ### Input preprocessing
        x = embedding(data)
        ### Forward pass
        if option == 3: 
            x = bilstm(x)
        output = model(x, y_embedding)
        metric.cm_compute(output, tag)
        ### Loss computation
        ### As we do viterbi inference we have the best path and the target path
        ### Here the loss is just the count of the miss predictions.
        loss += torch.sum(output!=tag).float()/tag.shape[1] 
    scores = metric.get_scores()
    return loss/batch_idx, scores

def save_model(model, bilstm, embedding,y_embedding, option, suffix, train_embedding, file = "./models", train_loader=None, val_loader=None, test_loader=None, separated_train_loader=None):

    if train_embedding:
        temp = 1
    else:
        temp = 0
    if not os.path.exists(file):
        os.makedirs(file)    
    torch.save(model.state_dict(),file+"/model_{}_".format(option )+ suffix)
    if option == 3:
        torch.save(bilstm.state_dict(),file+"/bilstm_{}_".format(option)+ suffix)
    torch.save(embedding.embedding.state_dict(),file+"/embedding_{}_{}_".format(option, temp)+ suffix)
    torch.save(y_embedding.embedding.state_dict(),file+"/y_embedding_{}_{}_".format(option, temp )+ suffix)
    
    if train_loader is not None:
        torch.save(train_loader,file+"/trainset_{}_".format(option ))
        torch.save(test_loader,file+"/testset_{}_".format(option ))
        torch.save(val_loader,file+"/valset_{}_".format(option ))
        torch.save(separated_train_loader,file+"/separated_train_loader_{}_".format(option ))
    return

def load_model(model, bilstm, embedding, y_embedding, option, suffix, train_embedding, file = "./models", load_dataset=False):
    if train_embedding:
        temp = 1
    else:
        temp = 0
    
    model.load_state_dict(torch.load(file+"/model_{}_".format(option )+ suffix))
    if option == 3:
        bilstm.load_state_dict(torch.load(file+"/bilstm_{}_".format(option)+ suffix))
    else:
        bilstm = None
    embedding.embedding.load_state_dict(torch.load(file+"/embedding_{}_{}_".format(option, temp)+ suffix))
    y_embedding.embedding.load_state_dict(torch.load(file+"/y_embedding_{}_{}_".format(option, temp )+ suffix))
    if load_dataset:
        trainset = torch.load(file+"/trainset_{}_".format(option ))
        valset = torch.load(file+"/valset_{}_".format(option ))
        testset = torch.load(file+"/testset_{}_".format(option ))
        separated_train_loader = torch.load(file+"/separated_train_loader_{}_".format(option ))
        return model, bilstm, embedding, y_embedding, trainset, valset, testset, separated_train_loader
    else:
        return model, bilstm, embedding, y_embedding

def main():
    parser = argparse.ArgumentParser()
    ### Default parameters
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--test_predictions_file', type=str, default='data/test_predictions.csv', help='Test file')
    parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    
    ### training parameters
    parser.add_argument('--optim', type=str, default="adam", help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--embed_lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--optim', type=str, default="sgd", help='optimizer')
    # parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--embed_lr', type=float, default=1e-2, help=' learning rate')
    parser.add_argument('--lam', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum')
    parser.add_argument('--nesterov', default=False, action='store_true', help='To save the initial state for reproducability')
    parser.add_argument('--split', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='number of iterations')
    parser.add_argument('--load_init', default=False, action='store_true', help='To load the initial state for reproducability')
    parser.add_argument('--save_init', default=False, action='store_true', help='To save the initial state for reproducability')
    parser.add_argument('--train_embedding', default=False, action='store_true', help='To save the initial state for reproducability')
    parser.add_argument('--include_imbalance', default=False, action='store_true', help='To save the initial state for reproducability')
    parser.add_argument('--suffix', type=str, default="small", help='save suffix also decides how big the model is')
    
    ###model parameters
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch_size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch_size')
    parser.add_argument('--embed_size', type=int, default=300, help='learning rate')
    parser.add_argument('--y_embed_size', type=int, default=5, help='learning rate')
    parser.add_argument('--device', type=str, default="cpu", help='gpu device')
    parser.add_argument('--p', type=float, default=0.2, help='dropout fraction')
    parser.add_argument('--lstm_layer', type=int, default=2, help='Number of layers in LSTM')
    parser.add_argument('--vocab_size', type=int, default=15000, help='Vocab Size')
    parser.add_argument('--unk_size', type=int, default=20, help='Unknown Vocab size')
    args = parser.parse_args()

### Hyperparameters
    # if args.option==1:
    #     args.learning_rate = 1e-3
    #     args.embed_lr = 1e-3
    #     args.lam=1e-3
    #     args.epochs = 10

    # elif args.option == 2 :
    #     args.learning_rate = 1e-3
    #     args.embed_lr = 1e-3
    #     args.lam=1e-3
    #     args.epochs = 10
    # else:
    #     args.learning_rate = 1e-3
    #     args.embed_lr = 1e-3
    #     args.lam=1e-3
    #     args.epochs = 10
        
    raw_tag_2_label = {
        "O":0,
        "T-POS":1,
        "T-NEU":2,
        "T-NEG":3
    }
    label_2_raw_tag = {
        0:"O",
        1:"T-POS",
        2:"T-NEU",
        3:"T-NEG"
    }   
    

###Initialize all the embeddings
    ### Embeddings of the input and y 
    embedding = Embedding(args.embed_size, max_vocab_size = args.vocab_size, unk_words = args.unk_size, device= args.device).to(args.device)
    y_embedding = Embedding(args.y_embed_size, max_vocab_size = args.num_classes+1, unk_words = 0, device= args.device).to(args.device)
    if args.option == 3:
        bilstm=BiLSTM(embed_dim = args.embed_size, hidden_dim =  args.embed_size//2, lstm_layer= args.lstm_layer, p=args.p, device = args.device, suffix= args.suffix)
    else:
        bilstm = None
    file = "./save_init/opt_{}".format(args.option)
    if args.train_embedding:
        file+="_1"

    if args.load_init:
    ###Initialize all the models 
        test_set = read_data(path=args.test_file)
        ###Initialize the Prediction and inference model
        model = TargetedSentiment(num_classes = args.num_classes, batch_size=args.test_batch_size, embed_dim=args.embed_size, y_embed_dim=args.y_embed_size, p=args.p, device=args.device, imbalance=None, suffix= args.suffix).to(args.device)
        model, bilstm, embedding,y_embedding, train_loader, val_loader,test_loader, separated_train_loader = load_model(model, bilstm, embedding, y_embedding,args.option, args.suffix, args.train_embedding, file, True)
    
    else:

        # read the dataset
        train_set = read_data(path=args.train_file)
        test_set = read_data(path=args.test_file)
        random.shuffle(train_set)
        val_set = train_set[:int(args.split * len(train_set))]
        train_set = train_set[int(args.split * len(train_set)):]
        
        ### Word2Vec
        if args.option==2:
            print("Loading the w2v model")
            # wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
            wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/local/scratch/a/skodge/min/Projects/CS-57700/hw2/word2vec/w2v.bin"), binary=True)
        else:
            wv_from_bin=None


    ###Process training and testing data    
        train_loader, separated_train_loader = train_preprocess(train_set, embedding, wv_from_bin, raw_tag_2_label, args.option, args.device)
        test_loader = test_preprocess(test_set, embedding, wv_from_bin, raw_tag_2_label, args.option, args.device)
        val_loader = test_preprocess(val_set, embedding, wv_from_bin, raw_tag_2_label, args.option, args.device)

        model = TargetedSentiment(num_classes = args.num_classes, batch_size=args.test_batch_size, embed_dim=args.embed_size, y_embed_dim=args.y_embed_size, p=args.p, device=args.device, imbalance = None, suffix= args.suffix).to(args.device)
    
    ### Save or load the models
        if args.save_init:
            save_model(model, bilstm, embedding,y_embedding, args.option, args.suffix, args.train_embedding,file, train_loader, val_loader,test_loader, separated_train_loader)
## As the dataset is hugely baised towards class 0
    if args.include_imbalance:
        imbalance= []
        for class_num in separated_train_loader:
            imbalance.append(len(separated_train_loader[class_num]))
    else:
        imbalance = None 
    model.scale_func(imbalance)
        
 
    if args.option==2 and args.train_embedding :
        embedding.embedding.weight = torch.nn.Parameter(embedding.embedding.weight.clone().detach().to(args.device))
        embedding.embedding.requires_grad = True
    
    if args.option==2 and not args.train_embedding:   
        embedding.embedding.weight = torch.nn.Parameter(embedding.embedding.weight.clone().detach().to(args.device))
        embedding.embedding.requires_grad = False

### params for the optimizer  
    params = [
                    {'params': model.parameters()},
                    {'params': y_embedding.parameters(), "lr": args.embed_lr},
    ]
    if args.option == 3:
        params.append({'params': bilstm.parameters()})
    if args.option !=2 or args.train_embedding:
        params.append({'params': embedding.parameters(), "lr": args.embed_lr})

###initialize the optimizer
    if args.optim =="sgd":
        optimizer = optim.SGD(params, lr=args.learning_rate,momentum=args.momentum, weight_decay=args.lam, nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.lam)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

### loss function
    criterion = nn.CrossEntropyLoss()
    
    ###Metric(Compute F1 Recall Precision and accuracy)
    metric = metric_computations(args.num_classes)
    
#### Training
    print("Training")
    file = "./models/opt_{}_lr_{}_lam_{}".format(args.option,args.learning_rate, args.lam)
    if args.train_embedding:
        file+="_1"
    if not os.path.exists(file):
        os.makedirs(file)    
    
    model, bilstm, embedding, y_embedding, optimizer, scheduler, train_loss, validation_loss, test_loss = train(model, 
                                                                                                                bilstm,
                                                                                                                embedding, 
                                                                                                                y_embedding, 
                                                                                                                train_loader, 
                                                                                                                val_loader,
                                                                                                                test_loader,
                                                                                                                args.epochs, 
                                                                                                                criterion, 
                                                                                                                optimizer,
                                                                                                                scheduler,
                                                                                                                args.batch_size, 
                                                                                                                metric, 
                                                                                                                args.device,
                                                                                                                args.option,
                                                                                                                args.train_embedding,
                                                                                                                args.suffix,
                                                                                                                file)
### Load best saved model
    model, bilstm, embedding,y_embedding = load_model(model, bilstm, embedding,y_embedding, args.option, args.suffix, args.train_embedding,file )

### Testing trained parameters
    print("Testing after training")
    train_score = test(model, bilstm, embedding, y_embedding, train_loader, metric, args.option)
    val_score =test(model, bilstm, embedding, y_embedding, val_loader, metric, args.option)
    test_score =test(model, bilstm, embedding, y_embedding, test_loader, metric, args.option)
    print(train_score)
    print(val_score)
    print(test_score)   
    plt.plot(range(0, args.epochs+1,1), train_loss, label ="Training Curve-F1:{:.4}-Loss:{:.4}".format(train_score[1]["f1"],train_score[0]))
    plt.plot(range(0, args.epochs+1,1), validation_loss, label = "Validation Curve-F1:{:.4}-Loss:{:.4}".format(val_score[1]["f1"],val_score[0]))
    plt.plot(range(0, args.epochs+1,1), test_loss, label = "Test Curve-F1:{:.4}-Loss:{:.4}".format(test_score[1]["f1"],test_score[0]))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Option {} Learning Rate {} Weight Decay {}".format( args.option, args.learning_rate, args.lam))
    plt.legend()
    #plt.show()
    plt.savefig(file +"/learning_curve_{}_{}_{}_{}_".format(args.option, args.epochs, args.learning_rate, args.lam)+ args.suffix+".png")

# save the predictions
    test_set_with_labels = pd.DataFrame(test_set.copy())
   
    # TODO: change this portion to save your predicted labels in the dataframe instead, here we are just saving the true tags. Make sure the format is the same!
    test_set_with_labels['ts_pred_tags'] = test_set_with_labels['ts_raw_tags']
    
    for batch_idx, (data, tag, y_prev, idx) in enumerate(test_loader):
        ### Input preprocessing
        x = embedding(data)
        ### Forward pass
        output = model(x, y_embedding).cpu()
        ### update the vector to be saved
        test_set_with_labels['ts_pred_tags'][idx]= [label_2_raw_tag[tag.item()] for tag in output.squeeze(0)]

        
    # now, save the predictions in the file
    # you can change this function but make sure you do NOT change the format of the file that is saved
    SaveFile(test_set_with_labels, output_csv_file=args.test_predictions_file)
    _, scores = test(model, bilstm, embedding, y_embedding, test_loader, metric, args.option)
    print("")
    print("Accuracy: {}".format(scores["acc"]))
    print("Precision: {}".format(scores["precision"]))
    print("Recall: {}".format(scores["recall"]))
    print("F1: {}".format(scores["f1"]))
    
if __name__ == '__main__':
    main()   




