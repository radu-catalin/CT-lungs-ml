import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# device config
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
num_epochs = 25
batch_size = 100
learning_rate = 0.001
momentum = 0.9
no_filter1 = 20
no_filter2 = 50
no_neurons = 500
hidden_layer_linear = 50
log_interval = int(1000 / batch_size)

path_train = './kaggle/train'
path_train_metadata = './kaggle/train.txt'

path_validation = './kaggle/validation'
path_validation_metadata = './kaggle/validation.txt'

path_test = './kaggle/test'
path_test_metadata = './kaggle/test.txt'

def get_dataset(path_images: str, path_metadata: str, shuffle: bool = False, batch_size: int = batch_size) -> None:
	images_metadata = np.genfromtxt(path_metadata, delimiter=',', dtype='str')
	images_name = images_metadata[:, 0]

	np_dataset_images = np.array([np.array(Image.open(path_images + '/' + name)) for name in images_name])
	np_dataset_images = np.expand_dims(np_dataset_images, 1)

	np_dataset_labels = images_metadata[:, 1].astype(np.int32)

	dataset_images, dataset_labels = map(
		torch.tensor, (np_dataset_images, np_dataset_labels)
	)

	dataset_images = TensorDataset(dataset_images, dataset_labels)

	dataset_loader = DataLoader(
		dataset = dataset_images,
		batch_size = batch_size,
		shuffle = shuffle
	)

	return dataset_loader

def get_dataset_no_labels(path_images: str, path_metadata: str, shuffle: bool = False, batch_size: int = batch_size) -> None:
	images_metadata = np.genfromtxt(path_metadata, delimiter=',', dtype='str')
	images_name = images_metadata

	np_dataset_images = np.array([np.array(Image.open(path_images + '/' + name)) for name in images_name])
	np_dataset_images = np.expand_dims(np_dataset_images, 1)
	np_dataset_labels = np.array([0 for name in images_name])

	dataset_images, dataset_labels = map(
		torch.tensor, (np_dataset_images, np_dataset_labels)
	)

	dataset_images = TensorDataset(dataset_images, dataset_labels)

	dataset_loader = DataLoader(
		dataset = dataset_images,
		batch_size = batch_size,
		shuffle = shuffle
	)

	return dataset_loader

def preprocess(data) -> float:
  return data.float() / 255.0

def plot_loss(loss, label, color='red') -> None:
	plt.plot(loss, label=label, color=color)
	plt.legend()

train_loader = get_dataset(path_train, path_train_metadata)
validation_loader = get_dataset(path_validation, path_validation_metadata)
test_loader = get_dataset_no_labels(path_test, path_test_metadata, shuffle = False, batch_size = 1)

class ConvolutionalNetwork(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(
			in_channels = 1,
			out_channels = no_filter1,
			kernel_size = 5,
			stride = 1
		)

		self.pool = nn.MaxPool2d(2, 2)

		self.conv2 = nn.Conv2d(
			in_channels = no_filter1,
			out_channels = no_filter2,
			kernel_size = 5,
			stride = 1
		)

		self.fully_conv1 = nn.Conv2d(no_filter2, no_neurons, 3)
		self.fully_conv2 = nn.Conv2d(no_neurons, 5, 1)

		self.linear1 = nn.Linear(4050, hidden_layer_linear)
		self.linear2 = nn.Linear(hidden_layer_linear, 3)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# x = F.relu(self.fully_conv1(x))
		# x = self.fully_conv2(x)
		x = x.view(x.size(0), -1)
		x = F.relu(self.linear1(x))
		x = self.linear2(x)

		# return x
		return F.log_softmax(x, dim=1)

model = ConvolutionalNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

n_total_steps = len(train_loader)
losses_train = []
# training loop
for epoch in range(num_epochs):
	for i, (data, labels) in enumerate(train_loader):
		data = data.to(device).float()
		labels = labels.long()
		outputs = model(data)
		# loss = criterion(outputs, labels)
		loss = F.nll_loss(outputs, labels)
		# print(loss)

		# backward
		losses_train.append(loss.detach().cpu().numpy())
		loss.backward()

		# update
		optimizer.step()
		optimizer.zero_grad()

		if (i + 1) % log_interval == 0:
			print(
					f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
			)

print('Finished training!')

losses_validation = []
with torch.no_grad():
	n_correct = 0
	n_samples = 0

	for i, (data, labels) in enumerate(validation_loader):
		data = data.to(device).float()
		labels1 = labels.long()
		labels = labels.view(-1, 1).long()

		outputs = model(data)

		loss = F.nll_loss(outputs, labels1)
		# print(loss)

		# backward
		losses_validation.append(loss.detach().cpu().numpy())

		_, predicted = torch.max(outputs, dim=-1)
		predicted = predicted.view(-1, 1)
		n_samples += labels.size(0)
		n_correct += (predicted == labels).sum().item()

	print(n_samples, n_correct)
	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy of the network: {acc}%')

plot_loss(losses_train, 'loss')
plt.show()
plot_loss(losses_validation, 'loss')
plt.show()

f = open('./kaggle/submission.txt', 'w')
f.write('id,label\n')
with torch.no_grad():


	images_name = np.genfromtxt(path_test_metadata, delimiter=',', dtype='str')

	for i, (data, labels) in enumerate(test_loader):
		data = data.to(device).float()
		labels = labels.view(-1, 1).long()

		outputs = model(data)

		_, predicted = torch.max(outputs, dim=-1)
		predicted = predicted.view(-1, 1)
		# print(predicted == labels)
		f.write(f'{images_name[i]},{predicted.item()}\n')
	f.close()