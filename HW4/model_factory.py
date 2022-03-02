################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from image_caption_models import  ImageCaptionLSTM, ImageCaptionRNN
from vocab import Vocabulary

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    model = None
    if model_type == 'LSTM':
        model = ImageCaptionLSTM(hidden_size, embedding_size, vocab)
    elif model_type == 'vanilla':
        model = ImageCaptionRNN(hidden_size, embedding_size, vocab)
    return model

    
