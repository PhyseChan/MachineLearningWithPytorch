from MinstDataLoader import MnistReader
import torch.optim as optim
import torch.nn as nn
from LSTM import LSTM
import torch
from sklearn import metrics
import numpy as np

training_loader, test_loader = MnistReader().dataloader(batch_size_trian= 50)

epochs = 1
model = LSTM(28, 32,10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(training_loader):
        data, label = item
        data = torch.squeeze(data, dim=1)
        output = model(data)
        loss = criterion(output.squeeze(), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _ % 100 == 99:
            print(f'loss: {sum_loss.item() / 100:.2f}')
            sum_loss=0.0

model.eval()
test_predict_list = []
test_label_list = []
for _, item in enumerate(test_loader):
    data_test, label_test = item
    data_test = torch.squeeze(data_test, dim=1)
    pred = model(data_test)
    pred = torch.argmax(pred.squeeze(), dim=1)
    test_label_list += label_test.tolist()
    test_predict_list += pred.tolist()

accuracy = metrics.accuracy_score(test_predict_list,test_label_list)
precision = metrics.precision_score(test_predict_list,test_label_list,average='macro')
recall = metrics.recall_score(test_predict_list,test_label_list,average='macro')
f1 = metrics.f1_score(test_predict_list,test_label_list,average='macro' )
print("accuracy: {:f}".format(accuracy))
print("precision: {:f}".format(precision))
print("recall: {:f}".format(recall))
print("f1 score: {:f}".format(f1))


