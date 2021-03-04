import torch.optim as optim
import torch.nn as nn
import torch
from Utils.DataLoader import MnistReader
from AttentionModel import LstmAttentionOnData,LstmAttentionOnLstm,LstmAttention
import sklearn.metrics as metrics

batch_size_trian = 80
batch_size_test = 80
dataloader = MnistReader()
train_loader, test_loader = dataloader.getdataloader(batch_size_trian=batch_size_trian, batch_size_test=batch_size_test)

epochs = 5
a = torch.zeros(batch_size_trian, 128)
model = LstmAttentionOnData(28, 128, 10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.009)
for item in enumerate(model.parameters()):
    print(item[1].shape)
for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(train_loader):
        data, label = item
        data = data.squeeze().cuda()
        data = data.permute(1, 0, 2)
        output = model(data, a)
        loss = criterion(output, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss
        if _ % 10 == 9:
            print("loss:", sum_loss / 10)
            sum_loss = 0.0

model.cpu()
model.eval()
test_predict_list = []
test_label_list = []
a = torch.zeros(1, batch_size_test, 128)
for _, item in enumerate(test_loader):
    data, label = item
    data = data.squeeze()
    data = data.permute(1, 0, 2)
    pred = model(data, a)
    test_label_list += label.tolist()
    test_predict_list += torch.argmax(pred, dim=1).squeeze().tolist()

accuracy = metrics.accuracy_score(test_predict_list, test_label_list)
precision = metrics.precision_score(test_predict_list, test_label_list, average='macro')
recall = metrics.recall_score(test_predict_list, test_label_list, average='macro')
f1 = metrics.f1_score(test_predict_list, test_label_list, average='macro')
print("accuracy: {:f}".format(accuracy))
print("precision: {:f}".format(precision))
print("recall: {:f}".format(recall))
print("f1 score: {:f}".format(f1))
