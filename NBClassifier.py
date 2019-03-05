#################################################################
# Author: Sardor Taylakov
# Professor: Jessen Havill
# CS 401: Natural Language Processing
# Fall 2018, Denison University
#################################################################

import math, operator, nltk, csv
from nltk.corpus import movie_reviews
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist

#################################################################
# get_data: Getting train and test sets
#################################################################
def get_data():
	hotel_reviews = []

	with open('Hotel_Reviews.csv', 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		next(csv_reader) 					# skip first line
		for line in csv_reader:
			negative_review = line[6].strip()
			positive_review = line[9].strip()
			if len(negative_review.split()) > 7:
				tokens = nltk.word_tokenize(negative_review)
				punct_tokens = [tokens[0]] # do not add full stop before the first word in the review
				for i in range(1, len(tokens)):
				  	if tokens[i][0].isupper():
				  		punct_tokens.append('.')
				  	punct_tokens.append(tokens[i])
				punct_tokens.append('.')
				review_list = negate(punct_tokens)
				temp_tuple = (review_list, 'neg')
				hotel_reviews.append(temp_tuple)
			if len(positive_review.split()) > 7:
				tokens = nltk.word_tokenize(positive_review)
				punct_tokens = [tokens[0]]
				for i in range(1, len(tokens)):
				  	if tokens[i][0].isupper():
				  		punct_tokens.append('.')
				  	punct_tokens.append(tokens[i])
				punct_tokens.append('.')
				review_list = negate(punct_tokens)
				temp_tuple = (review_list, 'pos')
				hotel_reviews.append(temp_tuple)
			

	train_set_num = int(len(hotel_reviews) * 0.8)  	# 80% of reviews
	train_set = hotel_reviews[:train_set_num]    	# first 80%
	test_set = hotel_reviews[train_set_num:]		# last 20%
	classes = ['neg', 'pos']
	return train_set, test_set, classes

#################################################################
# get_test_set: Getting additional test set to test labels
#################################################################
def get_test_set():
	hotel_reviews = []
	with open('7282_1.csv', 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		next(csv_reader) 					# skip first line
		for line in csv_reader:
			review = line[14].strip()
			if len(review.split()) > 7:
				tokens = nltk.word_tokenize(review)
				review_list = negate(tokens)
				hotel_reviews.append(review_list)

	return hotel_reviews

#################################################################
# negate: takes a review as a list and returns tokens
#################################################################
def negate(negative_review):
	tokens = list(negative_review)
	newlist = []
	deletelist = []

	while len(tokens) != 0:
		for token in tokens:
			newlist.append(token.lower())
			deletelist.append(token)
			if token not in ['not', 'no', 'n\'t']:
				pass
			else:
				break

		for token in deletelist:
			tokens.remove(token)
		deletelist.clear()

		for token in tokens:
			deletelist.append(token)
			if token not in ['.', ',', ';', '!', 'and', 'but', 'because', '-']:
				newlist.append('not_' + token.lower())
			else:
				newlist.append(token)
				break

		for token in deletelist:
			tokens.remove(token)
		deletelist.clear()

	return newlist

#################################################################
# binary NB: not efficient, not used 
#################################################################
def binarize(rev):
	review = list(rev)
	result_list = []

	temp = [] # for one sent at a time

	while len(review) != 0:
		for token in review:
			temp.append(token)
			if token in ['.', '!', ';']:
				break

		for token in temp:
			review.remove(token)

		fDist = FreqDist(temp)
		for f in fDist:
			if fDist[f] > 1:
				fDist[f] -= 1

		for f in fDist:
			while fDist[f] > 0:
				result_list.append(f)
				fDist[f] -= 1

		temp.clear()


	return result_list

#################################################################
# effective binarization 
#################################################################
def newbinary(rev):
	review = list(rev)

	new_list = []
	deletelist = []
	add_again = []
	temp = [] # for one sentence at a time

	while len(review) != 0:
		for word in review:
			deletelist.append(word)
			if word in ['.', '!', ';']:
				new_list.append(word)
				break
			if word not in temp:
				new_list.append(word)
				temp.append(word)
			elif word not in add_again:
				add_again.append(word)
			else:
				new_list.append(word)
				temp.append(word)
				add_again.remove(word)

		for word in deletelist:
			review.remove(word)
		deletelist.clear()
		temp.clear()

	return new_list

#################################################################
# train: Defining a function for training the model
#################################################################
def train(trainingSet, classes):			# takes the training set (tuples containing reviews and classes) and classes
	logprior = {}							# dictionary to keep the prior probability of each class
	V = set()								# set to hold vocabulary
	bigdoc = {}								# dictionary to keep the whole text with a corresponding class
	loglikelihood = {}						# dictionary to store the likelihood of the document given the class

	most_frequent_words = ['the', 'and', 'was', 'to', 'a', 'in', 'room', 'of', 'very',
	'for', 'is', 'hotel', 'were', 'i', 'it', 'with',
	'we', 'on', 'at', 'had', 'from', 'that', 'have', 'as', 'you', 'so', 'are', 'be', 'our', 'my', 'rooms', 'this', 'all', 'they', 'there',
	'which', 'would', 'out', 'could', 'really', 'us', 'an', 'one', 'when', 's', 'too', 'only', 'bit', 'or',
	'just', 'get', 'by', 'if', 'more', 'been', 'walk', 'me', 'up', 'time', 'did', 'also',
	'can', 'day', 'even', 'about', '2', 'next', 'than', 'some', 'much', 'after',
	'quite', 'other', 'made', 'everything', 'london', 'will', 'back', '5',
	'around', 'what', 'door', 'floor', 'two', 'got', 'do', 'stayed', 'place', 'size', 'near', 'extremely', 'your', 'metro', 'minutes', '3', 'go', 'work', '4',
	'over', 'need', 'first', 'any', 'use', 'front', 'walking', 'enough', 'people', 'extra', 'asked', 'booked', 'has', 'central',
	'booking', 'away', 'again', 'outside', 'because', 'off', 'should', 'morning', '.', '!', ',', ':', ';']

	binarizedTrainingSet = []

	for tupl in trainingSet:
		review = newbinary(tupl[0]) 					# more effective binarization
		temp = (review, tupl[1])
		binarizedTrainingSet.append(temp)

	for tupl in binarizedTrainingSet:					# dealing with Vocabulary
		for token in tupl[0]:							# tupl[0] is a list of tokens
			if token not in most_frequent_words:					
				V.add(token)


	for c in classes:															# iterate through each class in classes
		Ndoc = len(binarizedTrainingSet)										# get the number of documents in D
		Nc = len([item for item in binarizedTrainingSet if item[1] == c])   	# get the number of documents from D in class c
		logprior[c] = float(math.log(float(float(Nc)/float(Ndoc)), 2))			# calculate logprior

		# bigdoc[c] <- assign the whole text from class c
		classtext = []
		mylist = [item for item in binarizedTrainingSet if item[1] == c]  	# retreiving only text from tuples 

		for i in mylist:
			classtext = classtext + i[0]									# appending reviews to a list

		bigdoc[c] = classtext												# assigning appended text to bigdoc[c]

		formatted_tokens = []										
		for token in bigdoc[c]:
			if token not in most_frequent_words:
				formatted_tokens.append(token)

		fDist = FreqDist(formatted_tokens)									# calculating the frequencies of formatted_tokens
		countNumerator = 0
		for word in V:
			if word in fDist: 							
				countNumerator = fDist[word]
			else:
				countNumerator = 0
			loglikelihood[(word, c)] = math.log(float(countNumerator + 1) / float(len(formatted_tokens) + len(V)),2)


	return logprior, loglikelihood, V


#################################################################
# test: Defining a function for testing a single document
#################################################################
def test(testDoc, logPrior, logLikelihood, classes, vocab):
	total = {}													# dictionary to hold the sum of probabilities for each class
	for c in classes:
		total[c] = logPrior[c]   								# getting the initial prior probability for the class
		tokens = list(testDoc)
		for w in tokens:
			if w in vocab:
				total[c] = total[c] + logLikelihood[(w, c)] 	# adding the probabilities to the total sum
	# return key with the largest value
	return max(total.items(), key=operator.itemgetter(1))[0]


#################################################################
# testCorpus: Defining a function for testing the whole corpus of movie reviews
#################################################################
def testCorpus(testSet, logPrior, logLikelihood, classes, vocab):
	precision = 0
	recall = 0
	accuracy = 0

	truePositives = 0
	falsePositives = 0

	trueNegatives = 0
	falseNegatives = 0

	for review in testSet:   														# iterating over each review to assign a class
		classValue = test(review[0], logPrior, logLikelihood, classes, vocab)		# calling the test function
		if review[1] == 'neg':
			if review[1] == classValue:
				trueNegatives += 1
			else:
				falseNegatives += 1
		else:
			if review[1] == classValue:
				truePositives += 1
			else:
				falsePositives += 1

	precision = float(truePositives) / float(truePositives + falsePositives)
	recall = float(truePositives) / float(truePositives + falseNegatives)
	accuracy = float(truePositives + trueNegatives) / float(truePositives + falsePositives + trueNegatives + falseNegatives)

	return precision, recall, accuracy

#################################################################
# moreTest: Defining a function for testing the whole corpus of movie reviews
#################################################################
def moreTest(testSet, logPrior, logLikelihood, classes, vocab):

	for review in testSet:   													# review here is already a list, no need for an index		
		classValue = test(review, logPrior, logLikelihood, classes, vocab)		# calling the test function
		if classValue == 'pos':
			labels, adjectives = tag(review)
			print("===================")
			print(labels)
			print(adjectives)


#################################################################
# pos tagging
#################################################################
def tag(rev):
	review = list(rev)
	labels = []				# list to store adjective-noun pairs
	adjectives = set()		# set for storing adjectives
	tagged_list = nltk.pos_tag(review)
	for tupl in tagged_list:
		if tupl[1] == 'JJ':
			adjectives.add(tupl[0])
	bgs = list(nltk.bigrams(tagged_list))
	for bg in bgs:
		if (bg[0][1] == 'JJ' and bg[1][1] == 'NN') or (bg[0][1] == 'JJ' and bg[1][1] == 'NNS'):
			if not bg[1][0].startswith('not_'):
				labels.append(bg[0][0] + " " + bg[1][0])

	if len(labels) == 0 and len(adjectives) == 0:
		labels.append('good')
	else:
		for pair in labels:
			if pair.split()[0] in adjectives:
				adjectives.remove(pair.split()[0])

	return labels, adjectives


#################################################################
# main
#################################################################
def main():
	trainingSet, testSet, classes = get_data()
	logprior, loglikelihood, v = train(trainingSet, classes)
	precision, recall, accuracy = testCorpus(testSet, logprior, loglikelihood, classes, v)		# NOW TESTING THE MODEL WITH TEST SET
	print("precision = " + str(precision) + " recall = " + str(recall) + " accuracy = " + str(accuracy))

	reviews = get_test_set()
	moreTest(reviews, logprior, loglikelihood, classes, v)

if __name__ == '__main__':
	main()
