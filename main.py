#Seng 474 HW 1
#Jacob Lower
#V00819863

#We will label messages that predict what will happen in the future as class 1 
#and messages that contain a wise saying as class 0.


import math
import operator
import time
import collections

from stop_words import get_stop_words

#The Naive Bayes Classifier

testdata = open('data/testdata.txt',"r")
testlabels = open('data/testlabels.txt',"r")
traindata = open('data/traindata.txt',"r")
trainlabels = open('data/trainlabels.txt',"r")
results = open('data/results.txt',"w")
vocab = []
stop_words = get_stop_words('english')

countdocs = []
train_labels_numericals = []
#Key variables
number_sentences_in_vocab = 0
num_wise = 0
num_pred = 0
P_wise = 0 #wise saying
P_pred = 0 #prediction
line_trainlabes = [] #csv of trainlables
line_traindata = [] #csv of traindata
condprob_wise = [] # conditional probability relative to vocab for wise is true
conditonalprob_wise = []
condprob_pred = []
conditionalprob_pred = []
train_sentences = []
train_c = []


#each hold the numierical conditional probability of each vocab word
v_p_a = [] #vocab predictive affinity
v_w_a = [] #vocab wise affinity
argmax_pred = []
argmax_wise = []
final_testlabels = []

pred_comp = 1
wise_comp = 1

#new ideas
all_the_words = []
count_of_each_vocab = []
p_cond_prob = []
w_cond_prob = []

###FINAL VARIABLES###
#vocab = []
#line_traindata = []

#training time
"""
1. Get P(c) for 0 and 1
2. Get P(t|c) for each term t
"""
def train():
	byline_trainlabes = trainlabels.readlines()
	size_trainlabes = len(byline_trainlabes)
	global line_trainlabes #class value list
	global P_wise 
	global P_pred
	global num_pred
	global num_wise
	global train_c
	
	for i in byline_trainlabes:
		if i != "\n":
			line_trainlabes.append(i.rstrip('\n'))

	for i in line_trainlabes:
		if i == "0":
			P_wise = 1+P_wise
			train_c.append(i)
		else:
			P_pred = 1+P_pred
			train_c.append(i)
	num_wise = P_wise
	num_pred = P_pred
	P_wise = P_wise/size_trainlabes
	P_pred = P_pred/size_trainlabes
	
			
def extract_vocab():
	d = traindata.readlines()
	s = len(d)
	curr = ""
	global vocab
	global train_sentences
	for line in d:
		
		curr = line.split(" ")
		count = 0
		for i in curr:
			i = i.rstrip('\n')
			curr[count] = i
			count = count + 1
		train_sentences.append(curr)
		for w in curr:
			if w not in stop_words:
				if w not in vocab:
					vocab.append(w)

#for each C wise
def vocab_wise_affinity():
	global num_wise
	global P_wise
	global condprob_wise
	global conditonalprob_wise
	
	global v_w_a
	vocab_count = 0
	
	i = 0
	while(i < num_wise):
		condprob_wise.append(0)
		conditonalprob_wise.append(0)
		i = i+1
	count = 0
	for sentence in train_sentences:
		if count == num_wise:
			break
		else:
			
			vocab_count = 0
			for word in vocab:
				if word in sentence:
					condprob_wise[count] = condprob_wise[count] + 1
					
				else:
					condprob_wise[count] = condprob_wise[count] + 0
				vocab_count = vocab_count + 1
			conditonalprob_wise[count] = (condprob_wise[count]+1)/(num_wise+2)
			count = count +1
					
		
			
	
	
		
#for each C pred
def vocab_pred_affinity():
	#pred first so go up to num_pred
	global num_pred
	global P_pred
	global condprob_pred
	global conditionalprob_pred
	global v_p_a
	
	global num_wise
	global P_wise
	global condprob_wise
	global conditonalprob_wise
	
	
	i = 0
	while(i < len(vocab)):
		v_p_a.append(0)
		v_w_a.append(0)
		i = i +1
	v_count = 0 
	s_count = 0
	for sentence in train_sentences:
		v_count = 0
		for word in vocab:
			
			if word in sentence:
				if s_count<num_pred: #add to pred value
					v_p_a[v_count] = v_p_a[v_count] + 1
				else:
					v_w_a[v_count] = v_w_a[v_count] + 1
			v_count = v_count + 1
		
		s_count = s_count + 1
	count = 0
	for val in v_p_a:
		v_p_a[count] = (v_p_a[count]+1)/(num_pred+2)
		count = count + 1
	count = 0
	for val in v_w_a:
		v_w_a[count] = (v_w_a[count]+1)/(num_wise+2)
		count = count + 1
	"""
	while(i < num_pred):
		condprob_pred.append(0)
		conditionalprob_pred.append(0)
		i = i+1
	count = 0
	for sentence in train_sentences:
		if count == num_pred:
			break
		else:
			for word in vocab:
				if word in sentence:
					condprob_pred[count] = condprob_pred[count] + 1
					
				else:
					condprob_pred[count] = condprob_pred[count] + 0
			conditionalprob_pred[count] = (condprob_pred[count]+1)/(num_pred+2)
			count = count +1
"""
#####TEST FUNCTIONS START #####
vocab_test = []
test_sentences = []
score_pred = []
score_wise = []
test_prediction = []
final_prediction_numeric = []

def test_vocab():
	d = testdata.readlines()
	s = len(d)
	curr = ""
	global vocab_test
	global test_sentences
	for line in d:
		
		curr = line.split(" ")
		count = 0
		for i in curr:
			i = i.rstrip('\n')
			curr[count] = i
			count = count + 1
		test_sentences.append(curr)
		for w in curr:
			if w not in stop_words:
				if w not in vocab_test:
					vocab_test.append(w)
	
	
#for each c in C (so wise or pred)
def pred_test():
	global score_pred
	
	
	Sentence_count = 0
	vocab_count = 0
	for sentence in test_sentences: #extract sentence
		vocab_count = 0
		score_pred = math.log2(P_pred)
		
		for word in vocab:#for each word in vocab
			if word in sentence:
				score_pred = score_pred + math.log2(v_p_a[vocab_count])
			vocab_count = vocab_count + 1
		argmax_pred.append(score_pred)
		Sentence_count = Sentence_count + 1
		

def wise_test():
	global score_wise
	
	
	Sentence_count = 0
	vocab_count = 0
	for sentence in test_sentences: #extract sentence
		vocab_count = 0
		score_wise = math.log2(P_wise)
		
		for word in vocab:#for each word in vocab
			if word in sentence:
				score_wise = score_wise + math.log2(v_w_a[vocab_count])
			vocab_count = vocab_count + 1
		argmax_wise.append(score_pred)
		Sentence_count = Sentence_count + 1
	

def compare_argmax():
	count = 0
	global argmax_pred
	global argmax_wise
	global test_prediction
	PE = 0
	pred_chance = 0
	wise_chance = 0
	while(count < len(argmax_pred)):
		test_prediction.append(0)
		PE = argmax_pred[count] + argmax_wise[count]
		pred_chance = argmax_pred[count]/PE
		wise_chance = argmax_wise[count]/PE
		
		if(pred_chance > wise_chance):
			test_prediction[count] = 1
		else:
			test_prediction[count] = 0
		count = count+1
		

def the_final_test():
	global testlabels
	lines_testlabels = testlabels.readlines()
	global final_testlabels
	for num in lines_testlabels:
		if num != "\n":
			final_testlabels.append(num.rstrip('\n'))
	count = 0
	for num in final_testlabels:
		if num == '1':
			final_testlabels[count] = 1
		else:
			final_testlabels[count] = 0
		count = count + 1

###count each vocab
def count_of_all_terms():
	global all_the_words
	global count_of_each_vocab
	global number_sentences_in_vocab
	sentence_count = 0
	for sentence in train_sentences:
		for word in sentence:
			all_the_words.append(word)
		sentence_count = sentence_count + 1
	number_sentences_in_vocab = sentence_count
	count_of_each_vocab = collections.Counter(all_the_words)
	

def real_condprob():
	global count_of_each_vocab
	global vocab
	global p_cond_prob
	global w_cond_prob
	count = 0
	for word in vocab:
		if count < num_pred:
			count_of_each_vocab[word] = count_of_each_vocab[word]
		else:
			count = count + 1

			
			
###################################################################################			
final_vocab = vocab#Vocab 
total_docs = num_pred + num_wise #Count Docs 
all_pred_sentences = []
all_wise_sentences = []
all_pred_word_count = []
all_wise_word_count = []
Pred_counter = []
Wise_counter = []
conditional_probability_of_pred = []
conditional_probability_of_wise = []
def prime_training():
	global final_vocab
	global total_docs
	global num_pred
	global num_wise
	global count_of_each_vocab
	#to be made global
	global all_pred_sentences 
	global all_wise_sentences 
	global Pred_counter
	global Wise_counter
	#cound docs in each class
	class_count = 0
	for sentence in train_sentences:
		if class_count < num_pred:
			all_pred_sentences.append(sentence)
			for word in sentence:
				all_pred_word_count.append(word)
		else:
			all_wise_sentences.append(sentence)
			for word in sentence:
				all_wise_word_count.append(word)
		class_count = class_count + 1
	Pred_counter = collections.Counter(all_pred_word_count)
	Wise_counter = collections.Counter(all_wise_word_count)
	
	#for each t in V
	count = 0
	for t in vocab:
		conditional_probability_of_pred.append(0)
		conditional_probability_of_wise.append(0)
		conditional_probability_of_pred[count] = (1+Pred_counter[t])/(1+count_of_each_vocab[t])
		conditional_probability_of_wise[count] = (1+Wise_counter[t])/(1+count_of_each_vocab[t])
		count = count + 1

#### more globals
pred_score = []
wise_score = []
total_score = []
def apply_multinomial():
	global final_vocab
	global total_docs
	global num_pred
	global num_wise
	global count_of_each_vocab
	#to be made global
	global all_pred_sentences 
	global all_wise_sentences 
	global Pred_counter
	global Wise_counter
	global vocab_test	#vocab
	global test_sentences #extratec sentences
	global conditional_probability_of_pred
	global conditional_probability_of_wise
	global total_score
	
	global pred_score
	global wise_score
	
	count = 0
	for sentence in test_sentences:
		pred_score.append(math.log2(P_pred))
		wise_score.append(math.log2(P_wise))
		for word in sentence:
			pred_score[count] = pred_score[count] + math.log2((1+Pred_counter[word])/(1+count_of_each_vocab[word]))
			wise_score[count] = wise_score[count] + math.log2((1+Wise_counter[word])/(1+count_of_each_vocab[word]))
		
		
		count = count + 1
		
		
	for i in pred_score:
		total_score.append(i)
	counter = 0
	for i in wise_score:
		if total_score[counter] < wise_score[counter]:
			total_score[counter] = 0
		else:
			total_score[counter] = 1
		
		counter = counter + 1
	
	print("it is length: ",len(total_score))
			
		
	
			
###################################################################################		
#main trainable redo
pred_word_affinity = []
wise_word_affinity = []
train_pred_score = []
train_wise_score = []
train_total_score = []
def main_train():
	global count_of_each_vocab
	global number_sentences_in_vocab
	global num_pred
	global num_wise
	
	global Pred_counter
	global Wise_counter
	
	global train_pred_score
	global train_wise_score
	
	count = 0
	for sentence in train_sentences:
		train_pred_score.append(math.log2(P_pred))
		train_wise_score.append(math.log2(P_wise))
		for word in sentence:
			train_pred_score[count] = train_pred_score[count] + math.log2((1+Pred_counter[word])/(1+count_of_each_vocab[word]))
			train_wise_score[count] = train_wise_score[count] + math.log2((1+Wise_counter[word])/(1+count_of_each_vocab[word]))
		
		count = count + 1
	
	for i in train_pred_score:
		train_total_score.append(i)
	counter = 0
	for i in train_wise_score:
		if train_total_score[counter] < train_wise_score[counter]:
			train_total_score[counter] = 0
		else:
			train_total_score[counter] = 1
		
		counter = counter + 1
	
	print("it is length: ",len(train_total_score))
	
###################################################################################

def main():
	train()
	extract_vocab()
	count_of_all_terms()
	vocab_wise_affinity()
	vocab_pred_affinity()
	real_condprob()
	global results
	
	
	
	#got so far, conditionalprob_pred, conditonalprob_wise
	#trainning is done
	# C = num_pred + num_wise, V=vocab, prior = P_wise, P_pred, 
	# condprob = conditionalprob_pred, conditonalprob_wise
	
	test_vocab()
	pred_test()
	wise_test()
	
	compare_argmax()
	
	the_final_test()
	
	
	percentage_correct = 0
	count = 0
	#pred_score
	#wise_score
	
	for i in test_prediction:
		if final_testlabels[count] == test_prediction[count]:
			percentage_correct = percentage_correct + 1
		count = count + 1
	#print(final_prediction_numeric)
	percentage_correct = percentage_correct/len(test_prediction)
	#print("Percentage correct for this test is: ",percentage_correct)
	
	
	resulting_value = ("Percentage correct for this test is: " + str(percentage_correct))
	results.write(resulting_value)
	#print(train_sentences)
	#### NEW 
	prime_training()
	#print(count_of_each_vocab)
	apply_multinomial()
	#print(pred_score)
	
	#print("NEW prediction")
	count = 0
	#print(total_score)
	for i in test_prediction:
		if total_score[count] == test_prediction[count]:
			percentage_correct = percentage_correct + 1
		count = count + 1
	print(final_prediction_numeric)
	percentage_correct = percentage_correct/len(test_prediction)
	print("Percentage correct for this test is: ",percentage_correct)
	
	#print(count_of_each_vocab)
	main_train()
	global train_labels_numericals
	for i in line_trainlabes:
		train_labels_numericals.append(int(i))
	#print(len(train_labels_numericals))
	count = 0
	percentage_correct = 0
	for i in train_labels_numericals:
		if train_total_score[count] == train_labels_numericals[count]:
			percentage_correct = percentage_correct + 1
		count = count + 1
	print(percentage_correct/count)
	
#START
main()
	
	
	
		
	