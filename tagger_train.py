import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

torch.manual_seed(1)
all_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"{}#*=@`'


def accuracy(x, y):
    correct = 0.
    for i in range(len(x)):
        if x[i] == y[i]:
            correct += 1.
    return correct/len(x)


def prepare_sequence(seq, to_ix):
    idxs = list()
    for word in seq:
        # print(word)
        if to_ix.get(word) is not None:
            idxs.append(to_ix[word])
        else:
            idxs.append(random.randrange(len(to_ix)))

    return torch.tensor(idxs, dtype=torch.long)


def read_data(path):
    f = open(path, "r")
    training_data = []
    tags_set = set()
    for line in f:
        sentence = line.split()
        words = []
        tags = []
        for item in sentence:
            splitted = item.split("/")
            if len(splitted) > 2:
                for i in range(len(splitted)-1):
                    words.append(splitted[i])
                    tags.append(splitted[-1])
            else:
                words.append(splitted[0])
                tags.append(splitted[1])
        for tag in tags:
            tags_set.add(tag)
        training_data.append(tuple([words, tags]))
    return training_data, tags_set


def get_tag(sent, output, idx_tag):
    max_idxs = output.argmax(1).numpy()
    output = ""
    for i in range(len(sent)):
        output += sent[i] + "/" + idx_tag[max_idxs[i]] + " "
    return output.strip()


class LSTMTagger(nn.Module):

    def __init__(self, word_emb_dim, char_emb_dim, k, hidden_dim, vocab_size, tagset_size, l, word_len=54, sent_len=120):
        super(LSTMTagger, self).__init__()
        self.sent_len=sent_len
        self.hidden_dim = hidden_dim
        self.k = k
        self.tagset_size = tagset_size
        self.maxlen = word_len
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim)
        self.char_embeddings = nn.Embedding(len(all_chars), char_emb_dim)
        self.conv1d = nn.Conv1d(char_emb_dim, l, k, padding=int((k-1)/2))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.nonlinearity = nn.ReLU(inplace=False)
        self.lstm = nn.LSTM(word_emb_dim + l, hidden_dim, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*hidden_dim, self.tagset_size)


    def forward(self, sentence, chars):
        # sentence = [word_to_ix(w) for w in sentence]
        # sentence = sentence.view(1, self.sent_len, -1)

        #print('initial words_len:')
        #print(sentence.shape)

        word_embeds = self.word_embeddings(sentence)
        char_embeds = self.char_embeddings(chars)

        #print('chars_embeds:')
        #print(char_embeds.shape)

        conv = self.conv1d(char_embeds.permute(0, 2, 1))

        #print('conv:')
        #print(conv.shape)

        conv = self.nonlinearity(conv)
        #max_c = self.maxpool(conv)
        max_c = torch.max(conv, dim=2)[0]
        #print('maxpool')
        #print(max_c.shape)

        #print('word_embs')
        #print(word_embeds.shape)

        #max_c = max_c.permute(0, 2, 1)  # 141, 1, 6 -> 1, 141, 1, 6
        #max_c = max_c.view(max_c.shape[0], max_c.shape[2])

        #print('max_c:')
        #print(max_c.shape)

        input = torch.cat((word_embeds, max_c), dim=1)

        #print('lstm input:')
        #print(input.shape)

        # fcking shit is below:
        lstm_out, _ = self.lstm(input.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #print('before log_softmax')
        #print(tag_space[:,:10])
        tag_scores = F.log_softmax(tag_space, dim=1)
        #print('after log_softmax')
        #print(tag_scores[:,:10])
        return tag_scores

def train():
    ctraining_data, tags_set = read_data("corpus.train")
    word_to_ix = {}
    maxlen = -float('inf')
    max_sent = -float('inf')
    lengths = list()
    for sent, tags in ctraining_data:
        for word in sent:
            if len(word) > maxlen:
                maxlen = len(word)
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        if len(sent) > max_sent:
            max_sent = len(sent)
        lengths.append(len(sent))
    ctraining_data.sort(key=lambda x: len(x[0]), reverse=True)

    training_data = list()
    for sent, _ in ctraining_data:
        sentence = list()
        for word in sent:
            sentence.append(word + '#'*(maxlen - len(word)))
        training_data.append([sentence, _])


    # print(word_to_ix)
    # print(tags_set)

    tag_to_ix = {}
    ix_to_tag = {}
    count = 0
    for tag in tags_set:
        tag_to_ix[tag] = count
        ix_to_tag[count] = tag
        count += 1

    EMBEDDING_DIM = 17
    HIDDEN_DIM = 51
    CHAR_DIM = 19
    WINDOW_SIZE = 25
    L = 45



    model = LSTMTagger(EMBEDDING_DIM, CHAR_DIM, WINDOW_SIZE, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), L)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    initial_size = list()
    for index, element in enumerate(training_data):
        # new_sent = element[0]
        # for i in range(max_sent - len(element[0])):
        #     new_sent.append('#'*54)
        #     training_data[index][1].append('GGWP')
        initial_size.append(len(element[0]))
        # training_data[index][0] = new_sent

    for epoch in range(3):
        running_loss = 0.0
        counter = 0.0
        avg_loss = 0.0
        for sentence, tags in training_data:
            # clear gradient from the previous sentence
            model.zero_grad()

            # data preparation for the network
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            input_char = torch.tensor([[all_chars.index(c) for c in word] for word in sentence])

            # run forward pass
            tag_scores = model(sentence_in, input_char)

            # compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            # print out average loss for the last 2000 sentences
            running_loss += loss.item()
            counter += 1
            if counter % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, counter, running_loss / 2000))
                avg_loss += running_loss
                running_loss = 0.0
        print('average loss on epoch {} is {}'.format(epoch, avg_loss/counter))

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        input_char = torch.tensor([[all_chars.index(c) for c in word] for word in training_data[0][0]])
        print(np.array(training_data).shape)
        tag_scores = model(inputs, input_char)

        answers = np.argmax(np.array(tag_scores), axis=1)
        tags = [ix_to_tag[answer] for answer in answers]
        print(tags)
        print(training_data[0][1])
        print(accuracy(tags, training_data[0][1]))

    torch.save((word_to_ix, ix_to_tag, maxlen, model), 'model_file')

if __name__ == "__main__":
    train()
