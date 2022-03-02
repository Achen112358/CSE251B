import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.distributions.categorical import Categorical
from vocab import Vocabulary

class ImageCaptionLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab: Vocabulary):
        super().__init__()
        self.vocab = vocab
        vocab_size = len(self.vocab)

        self.encoder = torchvision.models.resnet50(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embedding_size)
        self.norm1 = nn.BatchNorm1d(embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.decoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, img, text):
        img_feat = self.encoder(img).unsqueeze(1)
        text_embedded = self.embedding(text[:,:-1])  # batch_size, seq_len, embedding_size
        x_input = torch.cat((img_feat,  text_embedded), dim=1)
        X_input = self.norm1(x_input.permute(0,2,1)).permute(0,2,1)
        decode, (_, _) = self.decoder(x_input)  # batch_size, seq_len, hidden_size
        out = self.output(decode).permute(0, 2, 1)  # batch_size, vocab_size, seq_len
        return out

    
    def forward_eval(self, img, generation_config):
        temperature = generation_config['temperature']
        max_length = generation_config['max_length']
        deterministic = generation_config['deterministic']

        batch_size = img.shape[0]

        img_feat = self.encoder(img).unsqueeze(1)
        text_lists = [['<start>'] for _ in range(batch_size)]
        decode, (hidden, cell) =  self.decoder(img_feat)

        while (not all([text_list[-1] == '<end>' for text_list in text_lists])) and all([len(text_list) <= max_length for text_list in text_lists]):
            text = torch.tensor([[self.vocab(text_list[-1])]
                                 for text_list in text_lists], dtype=torch.long).to(img.device)
                   
            text_embedded = self.embedding(text)  # batch_size, seq_len, embedding_size
            
            decode, (hidden, cell) = self.decoder(text_embedded, (hidden, cell))  # seq_len, batch_size, hidden_size
            out = self.output(decode).permute(0, 2, 1)  # batch_size, vocab_size, seq_len

            if deterministic:
                _, text_ids = F.log_softmax(out.squeeze_(), dim=1).max(dim=1)
            else:
                text_ids = Categorical(F.softmax(out.squeeze_() / temperature, dim=1)).sample()
            for text_list, text_id in zip(text_lists , text_ids):
                if text_list[-1] != "<end>":
                    text_list.append(self.vocab.idx2word[int(text_id.item())])

        text_lists = [[text for text in text_list if text != '<pad>' and text !=
                       '<start>' and text != '<end>' and text != '<unk>'] for text_list in text_lists]

        return text_lists
    
        
class ImageCaptionRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab: Vocabulary):
        super().__init__()
        self.vocab = vocab
        vocab_size = len(self.vocab)

        self.encoder = torchvision.models.resnet50(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embedding_size)
        self.norm1 = nn.BatchNorm1d(embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.decoder = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, nonlinearity='relu', batch_first=True)

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, img, text):
        img_feat = self.encoder(img).unsqueeze(1)
        text_embedded = self.embedding(text[:,:-1])  # batch_size, seq_len, embedding_size
        x_input = torch.cat((img_feat,  text_embedded), dim=1)
        X_input = self.norm1(x_input.permute(0,2,1)).permute(0,2,1)
        decode, _ = self.decoder(x_input)  # batch_size, seq_len, hidden_size
        out = self.output(decode).permute(0, 2, 1)  # batch_size, vocab_size, seq_len
        return out

    
    def forward_eval(self, img, generation_config):
        temperature = generation_config['temperature']
        max_length = generation_config['max_length']
        deterministic = generation_config['deterministic']

        batch_size = img.shape[0]

        img_feat = self.encoder(img).unsqueeze(1)
        text_lists = [['<start>'] for _ in range(batch_size)]
        decode, hidden =  self.decoder(img_feat)

        while (not all([text_list[-1] == '<end>' for text_list in text_lists])) and all([len(text_list) <= max_length for text_list in text_lists]):
            text = torch.tensor([[self.vocab(text_list[-1])]
                                 for text_list in text_lists], dtype=torch.long).to(img.device)
                   
            text_embedded = self.embedding(text)  # batch_size, seq_len, embedding_size
            
            decode, hidden = self.decoder(text_embedded, hidden)  # seq_len, batch_size, hidden_size
            out = self.output(decode).permute(0, 2, 1)  # batch_size, vocab_size, seq_len

            if deterministic:
                _, text_ids = F.log_softmax(out.squeeze_(), dim=1).max(dim=1)
            else:
                text_ids = Categorical(F.softmax(out.squeeze_() / temperature, dim=1)).sample()
            for text_list, text_id in zip(text_lists , text_ids):
                if text_list[-1] != "<end>":
                    text_list.append(self.vocab.idx2word[int(text_id.item())])

        text_lists = [[text for text in text_list if text != '<pad>' and text !=
                       '<start>' and text != '<end>' and text != '<unk>'] for text_list in text_lists]

        return text_lists
        