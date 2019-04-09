from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

dict_classes = { 0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

dict_indeces = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

length_dataset = len(y_test)
for i in range(0, length_dataset):
	target = y_test[i][0]
	dictionary[target] += 1
	dict_indeces[target].append(i)

for j in range(0, 10):
	print("CLASSE")
	print(dict_classes[j])
	print("INDICI")
	print(dict_indeces[j])
	print("NUM EXAMPLES")
	print(dictionary[j])