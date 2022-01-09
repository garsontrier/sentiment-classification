# sentiment-classification
Movie Review Sentiment Classification with Naive Bayes Classifiers

This is a mini project implemented in April 2020 for Introduction to Information Retrieval course in Bogazici University.

3 types of Naive Bayes classifiers (Multinomial, Bernoulli, and Binary) were implemented to carry out sentiment classification (positive/negative) on [Cornell Movie Review Dataset (polarity v2.0)](https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz). 


To run the code, please put the naive_bayes.py file in a directory which contains the data folder.
Dataset must be unzipped.
Here is an example directory:
```
Working dir
|
|------	naive_bayes.py
|	
|------	data

	|
	
	|------	train
	
	|	|
	
	|	|------	pos
	
	|	|	
	
	|	|------	neg
	
	|		
	
	|------	test
	
		|
		
		|------	pos
		
		|	
		
		|------	neg
		
 ```

 I've also attached auxilary .npy and .txt files to reduce run time. They should be put
 in the same directory as naive_bayes.py. Code works without those auxilary files but takes 
 considerably longer time.
 Code doesn't take any command line arguments.
