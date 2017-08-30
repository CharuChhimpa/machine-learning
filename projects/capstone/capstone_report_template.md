# Machine Learning Engineer Nanodegree
## Capstone Project
Charu Chhimpa   
August 23rd, 2017

## I. Definition

### Project Overview
Advancement in the field of medicine have greatly improved our quality of life which can be clearly seen from the life expectancy rate. From 1816’s stethoscope to today’s handheld ultrasound machines, doctors have steadily adopted technology to advance healthcare. In past decades the healthcare community has taken major steps by adopting electronic healthcare measures. Machine Learning can contribute significantly in the field of healthcare, it has great usage in cases like early
disease detection, finding signs of early breakouts of epidemics, using clustering to figure out regions of epidemics, or finding the best air quality zones in countries with high air pollution.  

Heart disease is the leading cause of death in the world over the past 10 years. The World Health Organization reported that heart disease is the first leading cause of death in high and low income countries. There are several methods in the literature individually to diagnosis diabetes or heart disease. There is no automated diagnosis method to diagnose Heart disease for diabetic patient based on diabetes diagnosis attributes to our knowledge. Researches have been using several data mining techniques in the diagnosis of heart disease. Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin, or when the body cannot effectively use the insulin it produces. Most of these systems have successfully employed Machine learning methods such as Naïve Bayes and Support Vector Machines for the classification purpose. Support vector machines are a modern technique in the field of machine learning and have been successfully used in
different fields of application.


### Problem Statement  
In this project we will train a model using Support Vector Machines to predict that a human being is suffering from a heart disease. We will consider several features to do this classification.  

**Task** : To predict that a person is suffering from a heart disease or not.  

**Performance** : Accuracy - No. of correct predictions.  

**Target Function** : A function that give the weights to every feature of a patient and then finally tell that the patient is sufering from heart disease or not. 

**Target Function Representation**: A Classification Model.


### Metrics
**Prediction Accuracy**  : The accuracy score will be used as an evaluation Metrics in this case. It would be calculated using the number of data points which are classified correctly.  
As it is a binary classification problem, prediction accuracy turns out to be the best evaluation metrics.   
If y_hat_i is the predicted value of the i-th sample and y_i is the corresponding true value, then the fraction of correct predictions over n_samples is defined as :   
                                
 ![](http://imgur.com/uzvxcmP.jpg)


## II. Analysis


### Data Exploration
I am using the UCI Heart Disease dataset for training the model. There are 14 features for every patient. In particular, there is data from 4 hospitals but I will use the Cleveland Database because it is properly processed. The dataset have 297 instances. The dataset link is : 
**http://archive.ics.uci.edu/ml/datasets/heart+Disease**   
Below is the image descibing the dataset.

![](http://imgur.com/g476e9q.jpg)


### Exploratory Visualization
The things that are explored in the previous section are :

#### Feature Distributions Compared for Disease and No Disease

![](http://imgur.com/l8KuwcC.jpg) 

![](http://imgur.com/JY012PQ.jpg)  

![](http://imgur.com/cIJ2peB.jpg) 



### Algorithms and Techniques
**Algorithm** : The Machine Learning Algorithm that we will useis Support Vecto Machines. 

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.  

The advantages of support vector machines are:
- Effective in high dimensional spaces.  

- Still effective in cases where number of dimensions is greater than the number of samples.  

- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.  

- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.  


### Benchmark
In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes’ theorem with strong (naive) independence assumptions between the features.

When dealing with continuous data, a typical assumption is that the continuous values associated with each class are distributed according to a Gaussian distribution. For example, suppose the training data contains a continuous attribute, x. We first segment the data by the class, and then compute the mean and variance of x in each class. Let µc be the mean of
the values in x associated with class c, and let σ^2_c be the variance of the values in x associated with class c. Suppose we have collected some observation value v. Then, the probability distribution of v given a class x, p(x=v), can be computed by plugging v into the equation for a Normal distribution parameterized by µc and σ^2_c. That is,  

![](http://imgur.com/4Un9Q5f.jpg)    

I’ll be using the Naive Bayes classifier as the benchmark model because it will always predict either of the one class.  

I’ll be looking to maximize the accuracy of the predictions using this model.


## III. Methodology

### Data Preprocessing
The features which are having more than two values are converted to dummy variables and given a different name for each variable. After converting the variables there are total of 18 variables.

### Implementation
First the Benchmark Model, Gaussian Naive Bayes is trained and it comes out to be with an accuracy of 78.33 %.  

After that Support Vector Machines are trained using the sklearn SVC and it came out with an accuracy of  85% . For increasing the accurcay we are Optimizing logistic regression on accuracy score, by using cross-validation. Here we use sklearn since it includes a cross_validation method. By using cross validation score we will find the set of features that yields the best accuracy score.  

Try eliminating features with a non-significant coefficient, one by one, while keeping the model deviance as low as possible. We'll use this second method for the final results.  

The final accuracy that we got after this is : 

### Refinement
The refinement that I made is that I used cross validation score to find the set of features that yields the best accuracy score .   

By using this we will eliminate features with a non - significant coefficient, one by one, while keeping the model deviance as low as possible. We will use this metthod for the final results. 


## IV. Results

### Model Evaluation and Validation
The final model came out with the accuracy of 85%  . The final features are selected on the basis of cross validation score. So, its a quite robust model tested properly.    

The initial model of SVMs is tuned properly to get a an accuracy of 85%.



### Justification
The final model performs really well in comparison of the Benchmark model. Our benchmark was to acheive atleast 78% accuracy and our model acheives nearly  85%  accuracy with a standard deviation of  0.008% . This clearly beats the benchmark model and is significant enough to solve the problem.

## V. Conclusion

### Reflection
The SVM model is properly trained and optimized using cross validation score. I faced a problem in slecting optimal model for this dataset then I tested it with several models and finally concluded on Support Vector Machines. The final accuracy of this model is quite good i.e.   


### Improvement
We can improve the model by further improving the parameters. We can even use grid search to get more optmized results.

-----------
### References 
- http://archive.ics.uci.edu/ml/datasets/heart+Disease  
- http://psrcentre.org/images/extraimages/84.%201211402.pdf  
- http://scikit-learn.org/stable/modules/svm.html
