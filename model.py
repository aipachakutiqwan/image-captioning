import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        

        self.hidden_size = hidden_size
        
        ## define embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        ## define the LSTM
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=num_layers, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )
        
        
        ## define the final, fully-connected output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    
    
    def init_hidden(self, batch_size):
        """ hidden state with all zeros (num_layers, batch_size, hidden_dim) """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),torch.zeros((1, batch_size, self.hidden_size), device=device))

    
    def forward(self, features, captions):
        ''' Forward pass through the network. 
        These inputs are features, and the hidden/cell state `captions`. '''
        
        captions = captions[:, :-1] 

        # Initialize the hidden state
        self.batch_size = features.shape[0] 
        self.hidden = self.init_hidden(self.batch_size) 
        
        # Create embedded word vectors
        embeddings = self.word_embeddings(captions) 

        # Stack the features and captions, embeddings new shape: (batch_size, caption length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # lstm_out shape: (batch_size, caption length, hidden_size)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) 

        # Fully connected layer (batch_size, caption length, vocab_size)
        outputs = self.linear(lstm_out) 

        return outputs
    
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "        
        output = []
        batch_size = inputs.shape[0] 
        hidden = self.init_hidden(batch_size) 
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out (1, 1, hidden_size)
            outputs = self.linear(lstm_out)  # outputs (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) 
            output.append(max_indice.cpu().numpy()[0].item()) 
            
            # We predicted the <end> word, break
            if (max_indice == 1):     
                break
            
            ## Prepare the new input of the lstm
            inputs = self.word_embeddings(max_indice) # inputs (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs (1, 1, embed_size)
            
        return output
        
        