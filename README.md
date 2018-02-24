Logistic Regression - 

Implementation of a Natural Language Processing System (Mini Siri), capable of processing flight information search terms using a Logistic Regression Model. 

The python script tagger.py implements a text analyzer using multinomial logistic regression. The script learns the parameters of a multinomial Logistic regression model that predicts a label for each word and its corresponsing feature vector. The program outputs the labels of the training and test examples and calculates training and test error.

Some conditions followed while implementation - 
1. All Model Parameters intialized to 0.
2. Model parameters optimized using stochastic gradient descent (SGD).  
3. Learning rate 'eta' is 0.5
4. The number of loops performed by the SGD is specified as a command line arg.
5. SGD is performed in the order in which the data is given. Although, typicall the input is shuffled, but here it is not.
6. To resolve ties where multiple classes have the same likelihood, the label with smaller ASCII value is chosen.

