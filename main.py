from nltk.stem.lancaster import LancasterStemmer
import json
import random
import tensorflow
import tflearn
import numpy
import nltk
import pickle
nltk.download('punkt')
stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)
# print(data['intents'])


try:
	with open ('data.pickle', 'rb') as f:
		word, labels, train, output=pickle.load(f)

except:
	#dont repeat
	words = []
	labels = []
	docs_x = []  # pattern
	docs_y = []  # tag

	for intent in data['intents']:
	    for pattern in intent['patterns']:
	        # each word to root
	        word = nltk.word_tokenize(pattern)
	        words.extend(word)
	        docs_x.append(word)
	        docs_y.append(intent['tag'])

	        if intent["tag"] not in labels:
	            labels.append(intent['tag'])


	# preprocess
	words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
	# remove duplicates, need list bc set its his own datatype
	words = sorted(list(set(words)))
	labels = sorted(labels)


	# onehotencoding
	train = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
	    bag = []
	    word = [stemmer.stem(w) for w in doc]  # stem each word
	    for w in words:
	        if w in word:
	            bag.append(1)
	        else:
	            bag.append(0)

	    output_row = out_empty[:]
	    # look in the labels list, see where the tag is and set the value to 1
	    output_row[labels.index(docs_y[x])] = 1
	    train.append(bag)
	    output.append(output_row)

	train = numpy.array(train)
	output = numpy.array(output)

	with open ('data.pickle', 'wb') as f:
		pickle.dump((word, labels, train, output), f)


tensorflow.reset_default_graph()
# each training input will be the same length
net = tflearn.input_data(shape=[None, len(train[0])])
net = tflearn.fully_connected(net, 8)  # hidden layer with 8 neturons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(
    net, len(output[0]), activation="softmax")  # proba
net = tflearn.regression(net)
model = tflearn.DNN(net)


try:
	model.load('model.tflearn')
except:
	# fitting
	model.fit(train, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")


#turn a sentence into a bag of words

def bag_of_words(s, words):
	bag =[0 for _ in range(len(words))] #bunch of 0 for how many words we have
	s_words=nltk.word_tokenize(s)
	s_words=[stemmer.stem(words.load()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se: #if the word we have is the word in the sentence
				#bag[i].append(1)
				bag[i]=1

	return numpy.array(bag)



#response
def chat():
	print("What do you have to say for yourself bruh? / type Q to stop")
	while True:
		inp=input("You: ")
		if inp.lower() == "Q":
			break

		result = model.predict([bag_of_words(inp, words)]) 
		#create a bag of words with the imput that we gave it
		#print(result)
		result_index=numpy.argmax(result) 
		# index of the greatest value in our list
		tag = labels[result_index]
		 #gives us the label that it thinks our message is 
		#print(tag)
		for tg in  data['intent']:
			if tg ['tag'] == tag:
				response=tg['response']

		print(random.choice(response))

chat()