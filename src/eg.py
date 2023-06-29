import snn

net = snn.default_snn(nn_in=3, nn_out=4, nn_hidden=5, dropout=0.01)
print(net)