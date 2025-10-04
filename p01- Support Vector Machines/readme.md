Daniel Kim HW #1 ReadMe

Note: My solutions for each question are labelled as a comment on the file hw1code.py

Question 3: Data Partitioning

My thought process was to make a universal data partitioning system for both MNIST and SPAM. I accomplished this by recalibrating the function so that when it recieves the number of data to put away for validation as a number, it would function to MNIST but if it is a decimal, then it would function to SPAM.

For the evaluation metric, I decided to approach it simply by find the mean of when the value equals the prediction, which was shown as a mathematical formula in the homework assignment

Question 4: Support Vector Machines: Coding

When I started this question, I coded the function for plotting the two data sets using Matlab and utilized the SVC(kernel="linear") as my SVM model. However, when i was doing this question, I wasn't really sure about what I was doing and created an elaborate function with a helped function that worked specifically for question 4. If I did this question again, I would simply make a funtion that would model for all parts of this coding homework assignment. Otherwise, I was able to calculate the training and validation accuracy of the MNIST data set and SPAM data set by running the training data/labels and validation data/labels  after using my data partition function. I then used my plot function to illustrate the spam and MNIST plots

Question 5: Hyperparameter Tuning

The C values I tried were: [1e-08, 5e-08, 1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-04]

I tried so many values for c, which ultimately cost me a slip day. I kept getting the same validation accuracy while using the recommended example c values. However, after I accidentally used a really small c value, I found that my code wasn't wrong but rather I was using the wrong c values. After ammending my c values and testing my code for the highest validation accuracy for the MNIST dataset, I found that it was around ~5e-07.

As for the code, I used the partitioned data and created a universal function that took the c value and was able to run it through the SVC(kernel="linear") SVM model which outputted the model that I used to predict the validation data and found the validation accuracy with the code.

Question 6: K-Fold Cross-Validation

The C Values I tried were: [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]

I tried a lot of values for c, ranging from very small to very large. I ultimately settled for C = 10, since that seemed to consistently give me the largest value for c, pretty much a similar pattern to Question 5 just for Spam. 

As for the code, I used the partitioned data and used the universal function that took the c value and was able to run it through the SVC(kernel="linear") SVM model which outputted the model that I used to predict the validation data and found the validation accuracy with the code.


Question 7: Kaggle

To keep increasing the value of the validation accuracy for SPAM, I kept adding a number of features that I found to be extremely present in SPAM emails. For example, upon investigating, I found that China and money were words that were used a lot in SPAM emails. While words like Money were already incorporated, I added words like China and free into my features that helped train and elevate the validation score of my code. For MNIST, my code was able to discern the test data at a very high clip ~95%, and required only my first attempt to pass the test. Words that didn't help that I tried were like reservation, which made me rethink a lot of words that I chose for features. Additionally phrases like "you win" hurt the validation score and I didn't realize this until removing all phrases from the features list. 