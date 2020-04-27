import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing, GraphConv, TopKPooling
from torch_geometric.utils import add_self_loops, remove_self_loops, degree
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pdb
import os , sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(save_path):
	'''
	read data from .npy file 
	no need to modify this function
	'''
	raw_data = np.load(save_path, allow_pickle=True)
	dataset = []
	for i, (node_f, edge_index, edge_attr, y)in enumerate(raw_data):
		sample = Data(
			x=torch.tensor(node_f, dtype=torch.float),
			y=torch.tensor([y], dtype=torch.float),
			edge_index=torch.tensor(edge_index, dtype=torch.long),
			edge_attr=torch.tensor(edge_attr, dtype=torch.float)
		)
		dataset.append(sample)
	return dataset


class GraphNet(MessagePassing):
	'''
	Graph Neural Network class
	'''
	def __init__(self, n_features):
		'''
		n_features: number of features from dataset, should be 37
		'''
		super(GraphNet, self).__init__()
		# define your GNN model here
		# raise NotImplementedError


		self.conv1 = GraphConv(n_features, 64)
		self.pool1 = TopKPooling(64, ratio=0.8)
		self.conv2 = GraphConv(64, 64)
		self.pool2 = TopKPooling(64, ratio=0.8)
		self.conv3 = GraphConv(64, 64)
		self.pool3 = TopKPooling(64, ratio=0.8)

		self.lin1 = torch.nn.Linear(128, 64)
		self.lin2 = torch.nn.Linear(64, 1)


	def forward(self, data):
		# x has shape [N, in_channels]
		# edge_index has shape [2, E]
		# pdb.set_trace()

		x          = data.x
		edge_index = data.edge_index
		batch      = data.batch
		edge_attr  = data.edge_attr

		x = F.relu(self.conv1(x, edge_index))
		x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
		x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = F.relu(self.conv2(x, edge_index))
		x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
		x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = F.relu(self.conv3(x, edge_index))
		x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
		x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = x1 + x2 + x3

		x = F.relu(self.lin1(x))
		x = self.lin2(x)

		x = x.reshape(-1)
		return x


def main():
	# load data and build the data loader
	
	train_set = get_dataset('train_set.npy')
	test_set = get_dataset('test_set.npy')
	train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

	save_dir = "./RESULTS"
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	# number of features in the dataset
	# no need to change the value
	n_features = 37

	# build your GNN model
	model = GraphNet(n_features)

	# define your loss and optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
	loss_func = torch.nn.MSELoss()

	print(model)

	hist = {"train_loss":[], "test_loss":[]}
	num_epoch = 100
	for epoch in range(1, 1+num_epoch):
		model.train()
		loss_all = 0
		for data in train_loader:
			# your codes for training the model
			optimizer.zero_grad()
			output = model(data)
			label = data.y
			loss = loss_func(output, label)
			loss.backward()
			optimizer.step()
			loss_all += loss.item() * data.num_graphs * len(data)
		train_loss = loss_all / len(train_set)

		with torch.no_grad():
			loss_all_test = 0
			for data in test_loader:
				# your codes for validation on test set
				optimizer.zero_grad()
				output = model(data)
				label = data.y
				loss_test = loss_func(output, label)
				loss_all_test += loss_test.item() * data.num_graphs * len(data)
			test_loss = loss_all_test / len(test_set)

			hist["train_loss"].append(train_loss)
			hist["test_loss"].append(test_loss)
			print("Epoch: [{}/{}], Train loss: {}, Test loss: {}".format(epoch, num_epoch, train_loss, test_loss))


	# test on test set to get prediction 
	with torch.no_grad():
		prediction = np.zeros(len(test_set))
		label = np.zeros(len(test_set))
		idx = 0
		for data in test_loader:
			data = data.to(device)
			output = model(data)
			prediction[idx:idx+len(output)] = output.squeeze().detach().numpy()
			label[idx:idx+len(output)] = data.y.detach().numpy()
			idx += len(output)
		prediction = np.array(prediction).squeeze()
		label = np.array(label).squeeze()

	loss_test = np.sum((label-prediction)**2)
	print("TestSetLossSum:", loss_test)

	# visualization
	# plot loss function
	ax = plt.subplot(1,1,1)
	ax.plot([e for e in range(1,1+num_epoch)], hist["train_loss"], label="train loss")
	ax.plot([e for e in range(1,1+num_epoch)], hist["test_loss"], label="test loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	ax.legend()
	# plt.show()
	plt.savefig(save_dir+"/loss.png")
	plt.close()

	# plot prediction vs. label
	x = np.linspace(np.min(label), np.max(label))
	y = np.linspace(np.min(label), np.max(label))
	ax = plt.subplot(1,1,1)
	ax.scatter(prediction, label, marker='+', c='red')
	ax.plot(x, y, '--')
	plt.xlabel("prediction")
	plt.ylabel("label")
	# plt.show()
	plt.savefig(save_dir+"/data.png")
	plt.close()

	# saving the model
	torch.save({
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	}, save_dir + '/model_epochnum_{}.ckpt'.format(epoch))


if __name__ == "__main__":
	main()
