# -*- coding: utf-8 -*-
"""
kNN Weighted & Unweighted CLassifier
Created on Thu Dec 1 09:00:00 2016
@author: Aman Dhingra
email: aman.dhingra@ucdconnect.ie
"""
from collections import Counter
import random

"""loadDataFile Loads the Data from matrix file in Simple Strings and Splits 
to generate 'myTrainSet' & 'myTestSet' with Random Documents of Split 70-30
	loadDataFile also loads Data from the Labels file passed and Stores it in List called 'labels'
	loadDataFile returns dictionaries 'myTrainSet','myTestSet' and List 'labels'"""
def loadDataFile(matrix_file,labelFile):
	matrix_file=open(matrix_file)
	i=0
	myTrainSet={}
	myTestSet={}
	#Populating Random Non-Repeating numbers in 'rows' from 1-1839. That is Document Numbers from 1-1839.
	rows=random.sample(range(1,1840),1839)
	#Reading Matrix File Line by Line
	for line in matrix_file.readlines():
		i=i+1
		#Leaving out the first 2 rows from file as they are Discription
		if(i>2):
			docNo,word,freq=line.strip().split(' ')
			#If docNo is in 70% of the documents in randomly populated 'rows' then myTrainSet insertion
			if(int(docNo) in rows[:int(len(rows)*0.7)]):
				if(int(docNo) not in myTrainSet.keys()):
					myTrainSet[int(docNo)]={}
				myTrainSet[int(docNo)][int(word)]=int(freq)
			#If docNo is in 30% of the documents in randomly populated 'rows' then myTestSet insertion
			else:
				if(int(docNo) in rows[int(len(rows)*0.7):]):
					if(int(docNo) not in myTestSet.keys()):
						myTestSet[int(docNo)]={}
					myTestSet[int(docNo)][int(word)]=int(freq)
	#For Labels file, splitting and storing in List 'labels'
	labels=[]
	labelFile=open(labelFile)
	for line in labelFile.readlines():
		n,label=line.strip().split(',')
		labels.append([int(n),label])
	#returns TrainingSet, TestSet and Labels list to main()
	return myTrainSet,myTestSet,labels
		
"""Calculates Cosine Similarity for each 'myTestSet' document against all documents in 'myTrainSet'.
	Calls unweightedClass() and weightedClass() to get predictions and check if predictions were Accurate
	Returns Booleans Recieved from the functions to main()"""
def cosineSim(myTrainSet,myTestDoc, myTestDocNo,labels,k):
	cosine_values={}
	#Computing Cosine Values for passed Test document against each Training Set document
	for doc in myTrainSet.keys():
		totalFrequency=0;
		totalFreqOfTrainDoc=0;
		totalFreqOfTestDoc=0
		#Common Words between the current document of Training Set and passed Test Set document
		CommonWords=(set(myTrainSet[doc].keys()) & set(myTestDoc.keys()))
		#Sum of Ai*Bi for all common words between A & B where A is the TestSet document and B is the TrainingSet document  
		for word in CommonWords:
			totalFrequency+=myTestDoc[word]*myTrainSet[doc][word]
		#Sum of Square of all word frequencies in TrainingSet Document
		for word in myTrainSet[doc].keys():
			totalFreqOfTrainDoc+=(myTrainSet[doc][word])**2
		#Sum of Square of all Word Frequencies in TestSet Document
		for word in myTestDoc.keys():
			totalFreqOfTestDoc+=(myTestDoc[word])**2
		#Cosine Similarity between the TestSet document and each TrainingSet document
		cosine_values[doc]=float((totalFrequency/(((totalFreqOfTrainDoc)**0.5)*((totalFreqOfTestDoc)**0.5))))
		#For Documents with highest Cosine Value 1, handling needs to be done so that Weighting avoids DivideByZero Error
		if (cosine_values[doc]==1.0):
			cosine_values[doc]=0.999
	#Returns Decisions from Predicted Vs Actual Labels by both Weighted and Unweighted Classifiers
	return unweightedClass(cosine_values,labels,myTestDocNo,k),weightedClass(cosine_values, labels,myTestDocNo,k)

"""Weighted kNN Classifier. Takes Document wise cosine Values as input with TestSet Document, k and labels List
	Predicts Class on basis of Weight=1/(1-CosineSimilarity).
	Checks if Predicted Class is Correct/Wrong and Returns Boolean for Correct/Wrong Prediction"""
def weightedClass(cosine_values, labels,myTestDocNo,k):
	votingGroup=Counter()
	for key in cosine_values.keys():
		distance=(1-cosine_values[key])
		cosine_values[key]=float(1/(distance))
	#Creating Sorted List of documents based on Weights in descending Order
	cosine_distance_sort=sorted(cosine_values, key=cosine_values.get, reverse=True)
	#Populating Voting Group from labels file for first k documents with highest Weights
	for element in range(0,k):
		votingGroup[labels[cosine_distance_sort[element]-1][1]]+=1
	#Sorting labels based on counts in descending Order
	topClass=sorted(votingGroup,key=votingGroup.get, reverse=True)
	#Checking if Predicted Class is Actual Class and returning Boolean
	#Subtracting 1 from myTestDocNo because labels list starts from 0
	if (topClass[0]==labels[int(myTestDocNo)-1][1]):
		return(True)
	else:
		return(False)

#Unweighted kNN CLassifier. Predicts class on basis of COsine Similarity Measure alone.
#Returns Result of Prediction.
def unweightedClass(cosine_values, labels,myTestDocNo,k):
	cosine_values_sort=sorted(cosine_values,key=cosine_values.get, reverse=True)
	votingGroup=Counter()
	#Subtracting 1 from cosine_values_sort[element] because labels list starts from 0
	for element in range(0,k):
		votingGroup[labels[cosine_values_sort[element]-1][1]]+=1
	topClass=sorted(votingGroup,key=votingGroup.get, reverse=True)
	#Checking if Predicted Class is Actual Class and returning Boolean
	
	if (topClass[0]==labels[int(myTestDocNo)-1][1]):
		return(True)
	else:
		return(False)
		
#Takes Value of k and Calls cosineSim function for each TestSet document. 
#Displays Accuracy fro both Weighted and Unweighted Classifier.
def main():
	matrix_file='../Data/news_articles.mtx'
	labelFile='../Data/news_articles.labels'
	k=int(input("Please Enter k Value: "))
	print("Loading Data Files...\nSpliting to TrainingSet and TestSet...")
	myTrainSet,myTestSet,labels=loadDataFile(matrix_file,labelFile)
	print("Data Files Loaded.")
	print("Calculating Accuracy on basis of Voting as well as Weights...")
	w_acc=Counter()
	u_acc=Counter()
	for myTestDocNo in myTestSet.keys():
		unweighted,weighted=cosineSim(myTrainSet, myTestSet[myTestDocNo],myTestDocNo,labels,k)
		w_acc[weighted]+=1
		u_acc[unweighted]+=1	
	print("For k = "+ str(k) +" Weighted: " +str(100*(float(w_acc[True])/float(len(myTestSet)))) + " Unweighted: " + str(100*(float(u_acc[True])/float(len(myTestSet)))))

main()
