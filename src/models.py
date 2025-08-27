import torch
import torch.nn as nn
import torch.nn.functional as F
from src.constants import *
torch.manual_seed(1)

class Layer(nn.Module):
	def __init__(
			self,
			hidden_dim,
			num_queries, # window
			activation_fn
			):
		super(Layer, self).__init__()

		self.activation_fn = activation_fn
		self.layer_norm = nn.LayerNorm(hidden_dim)

		self.dropout = nn.Dropout(0.1)

		self.learned_embed = nn.Embedding(num_queries, hidden_dim)
		nn.init.xavier_uniform_(self.learned_embed.weight)


	def forward(self, input):


		n, batch_size, feature_size = input.shape


		v_heads = input
		percetron_weight = self.learned_embed.weight.view(n, 1, feature_size)
		percetron_weight = percetron_weight.expand(-1, batch_size, -1) # n b h

		percetron_weight_norm = F.normalize(percetron_weight, p=2, dim=-1)
		v_heads_norm = F.normalize(v_heads, p=2, dim=-1)
		scores = torch.einsum("nbd,mbd->nbm", percetron_weight_norm, v_heads_norm)
		weights = F.softmax(scores, dim=-1)
		output = torch.einsum("nbm,mbd->nbd", weights, input) # 10 128 32
		output = self.dropout(output)

		return self.layer_norm(output + input)
	

class Perceptron(nn.Module):
	def __init__(self, feats):
		super(Perceptron, self).__init__()
		self.name = 'Perceptron'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window

		self.layer_num = 2   
		self.hidden_dim = 32
		self.input_proj = nn.Linear(self.n_feats, self.hidden_dim)
		self.pos_embed = nn.Parameter(torch.zeros(self.n_window, 1, self.hidden_dim))
		nn.init.normal_(self.pos_embed, std=0.02)
		self.Layers = nn.ModuleList([(Layer(hidden_dim=self.hidden_dim, num_queries=self.n_window, activation_fn=torch.nn.GELU())) for i in range (self.layer_num)])
		self.output_proj = nn.Linear(self.hidden_dim, self.n_feats)
		self.output_proj2 = nn.Linear(self.n_window, 1)

	def forward(self, src):
		x1 = self.input_proj(src)
		x1 = x1 + self.pos_embed
		for Layer in self.Layers:
			x1 = Layer(x1)
		x1 = self.output_proj(x1)
		x1 = self.output_proj2(x1.permute(2, 1, 0)).permute(2, 1, 0)
		return x1