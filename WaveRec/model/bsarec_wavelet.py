import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
from model._wavelet_family import WaveletFamily

class BSA_WaveRecModel(SequentialRecModel):
    def __init__(self, args):
        super(BSA_WaveRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) 
            print("all_encoder_layers : ", all_encoder_layers)
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha
        self.wavelet_layer = WaveletFilterLayer(args)

    def forward(self, input_tensor, attention_mask):
        wavelet_output = self.wavelet_layer(input_tensor) 
        gsp = self.attention_layer(input_tensor, attention_mask)        
        hidden_states = self.alpha * wavelet_output + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class WaveletFilterLayer(nn.Module):
    def __init__(self, args):
        super(WaveletFilterLayer, self).__init__()

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.pass_weight = args.pass_weight
        self.filter_type = args.filter_type  # haar, Daubechies(db2), coiflet, etc.
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

        # Initialize the WaveletFamily and generate filters
        wavelet_family = WaveletFamily(self.filter_type, self.pass_weight, args.filter_length, args.sigma)
        self.lowpass_filter, self.highpass_filter = wavelet_family.generate_filters()
    
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.transpose(1, 2)
        
        lowpass_filter = (self.lowpass_filter.view(1, 1, -1).repeat(input_tensor.size(1), 1, 1).to(input_tensor.device))
        highpass_filter = (self.highpass_filter.view(1, 1, -1).repeat(input_tensor.size(1), 1, 1).to(input_tensor.device))
        
        # 1D Convolution (applying the wavelet transform at the behavior level)
        with torch.no_grad():
            lowpass = F.conv1d(input_tensor, lowpass_filter, padding="same", groups=input_tensor.size(1))
            highpass = F.conv1d(input_tensor, highpass_filter, padding="same", groups=input_tensor.size(1))
        
        # Combine both results and restore the original dimensions
        wavelet_output = lowpass.transpose(1, 2) + (self.sqrt_beta ** 2) * highpass.transpose(1, 2)
        
        hidden_states = self.out_dropout(wavelet_output)
        hidden_states = self.LayerNorm(hidden_states + input_tensor.transpose(1, 2))  # Restore to the original dimensions
        
        return hidden_states
