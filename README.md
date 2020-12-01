# data-mining


Description of Implementation Process

Question 1

A.
We read the winequality-red.csv file and create a dataframe with it.
Then we divide the dataset into training set and set test with a ratio of 75% -25%.
In addition, with support vector machines we train the training set and try to
guess the quality of the wines of the test set.
Also, we try with grid search to find the combination of parameters C and
gamma that will give us the best categorization.
Finally, we print the parameters of the model that gave the best result
as well as the results of the metric f1 score, precision and recall for
categorization.

B.
1.
We read the winequality-red.csv file and create a dataframe with it.
Then we divide the dataset into training set and set test with a ratio of 75% -25%.
Also, we remove 33% of the values ​​of the ph column of the training dataset and we subtract it
column ph.
In addition, with support vector machines and the parameters that had the best
result in A we train the training set and try to guess
quality of the test set wines.
Finally, we print the results of the metric f1 score, precision and recall for
categorization.

2.
We read the winequality-red.csv file and create a dataframe with it.
Then we divide the dataset into training set and set test with a ratio of 75% -25%.
Also, subtract 33% of the values ​​of the ph column of the training dataset and complete
the values ​​with the average of the column items.
In addition, with support vector machines and the parameters that had the best
result in A we train the training set and try to guess
quality of the test set wines.
Finally, we print the results of the metric f1 score, precision and recall for
categorization.


3.
We read the winequality-red.csv file and create a dataframe with it.
Then we divide the dataset into training set and set test with a ratio of 75% -25%.
Also, subtract 33% of the values ​​of the ph column of the training dataset and complete
values ​​using Logistic Regression.
In addition, with support vector machines and the parameters that had the best
result in A we train the training set and try to guess
quality of the test set wines.
Finally, we print the results of the metric f1 score, precision and recall for
categorization.


4.
We read the winequality-red.csv file and create a dataframe with it.
Then we divide the dataset into training set and set test with a ratio of 75% -25%.
Also, subtract 33% of the values ​​of the ph column of the training dataset and complete
missing by the arithmetic mean of the cluster to which the sample belongs
applying k-means.
In addition, with support vector machines and the parameters that had the best
result in A we train the training set and try to guess
quality of the test set wines.
Finally, we print the results of the metric f1 score, precision and recall for
categorization.


Question 2

We read the onion-or-not.csv file and create a dataframe with it.
Next, we break the titles into words, creating a word vector
(tokenization) in line: df ['tokenized_sents'] = df.apply (lambda column:
nltk.word_tokenize (column ['text']), axis = 1).
Also, we remove their suffixes from the words, keeping only their subject
(stemming) in line: df ['stemmed'] = df ['tokenized_sents']. apply (lambda x: [ps.stem (y)
for y in x]).
In addition, we remove from the collection those words that are quite common and not
offer information (stopwords removal) on the line:
stop_words = set (stopwords.words ('english'))
stemmed.apply (lambda x: [item for item in x if item not in stop_words]).

Also, in the remaining words we assign as weight the tf-idf value in the line:
tfidf_vectorizer = TfidfVectorizer ()
dataframe = stemmed.to_frame ()
dataframe ['stemmed'] = ["" .join (review) for review in df ['stemmed']. values]
tfidf = tfidf_vectorizer.fit_transform (dataframe ['stemmed']).
Finally, we create a neural network to train and do
predict which title was published in the journal and which not.
