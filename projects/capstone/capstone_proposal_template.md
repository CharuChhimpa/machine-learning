# Machine Learning Engineer Nanodegree
## Capstone Proposal
Charu Chhimpa  
August 14th, 2017

### Domain Background
Advancement in the field of medicine have greatly improved our quality of life which can be clearly seen from the life expectancy rate. From 1816’s stethoscope to today’s handheld ultrasound machines, doctors have steadily adopted technology to advance healthcare. In past decades the healthcare community has taken major steps by adopting electronic healthcare measures. Machine Learning can contribute significantly in the field of healthcare, it has great usage in cases like early disease detection, finding signs of early breakouts of epidemics, using clustering to figure out regions of epidemics, or finding the best air quality zones in countries with high air pollution.  

Heart disease is the leading cause of death in the world over the past 10 years. Researches have been using several data mining techniques in the diagnosis of heart disease. Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin, or when the body cannot effectively use the insulin it produces. Most of these systems have successfully employed Machine learning methods such as Naïve Bayes and Support Vector Machines for the classification purpose. Support vector machines are a modern technique in the field of machine learning and have been successfully used in different fields of application. 




### Problem Statement

In this project we will train a model using Support Vector Machines to predict that a human being is suffering from a  heart disease. We will consider several features to do this classification.   

** Task **  : To predict that a person is suffering from a heart disease or not.  

** Performance ** : Accuracy - No. of correct predictions.  



### Datasets and Inputs

I am using the UCI Heart Disease dataset for training the model. This database contains 76 attributes, but the best results are obtained using a subset of 14 of them. In particular, there is data from 4 hospitals but I will use the Cleveland Database because it is properly processed.  

The different features that are used are  :   

1. age: age in years   

2. sex: sex (1 = male; 0 = female)   

3. cp: chest pain type   
-- Value 1: typical angina  
-- Value 2: atypical angina    
-- Value 3: non-anginal pain    
-- Value 4: asymptomatic   

4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)   

5. chol: serum cholestoral in mg/dl   

6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)   

7. restecg: resting electrocardiographic results   

8. thalach: maximum heart rate achieved  

9. exang: exercise induced angina (1 = yes; 0 = no)   

10. oldpeak = ST depression induced by exercise relative to rest   

11. slope: the slope of the peak exercise ST segment  
-- Value 1: upsloping   
-- Value 2: flat   
-- Value 3: downsloping   

12. ca: number of major vessels (0-3) colored by flourosopy   

13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect   

14. num: diagnosis of heart disease (angiographic disease status)  
-- Value 0: < 50% diameter narrowing   
-- Value 1: > 50% diameter narrowing   

### Solution Statement

I will use Supervised Learning approach to learn different features mentioned above and accordingly classify that the person is suffering from  heart disease or not. The dataset used is the Cleveland database which is pre processed before to seperate each of the 14 features by a comma. Supervised learning would finally provide us with binary output '0' or '1' based on the patient features.

Final Model that I would use is Support Vector Machines. Support vector machines are a modern technique in the field of machine learning and have been successfully used in different fields of application.


### Benchmark Model

#### Support Vector Machines   
SVM’s are very much useful in data classification. SVM’s classify data by finding an optimal hyper plane separating the d – dimensional data into its two classes with a maximum interclass margin. SVM’s use so called kernel functions to cast data into a higher dimensional space where the data is separable.   

SVM is a learning machine that plots the training vectors in high dimensional space and labels each vector by its class.
SVM based on the principle of risk minimization which aims to, minimize the error rate. SVM uses a supervised learning approach for classifying data. That is, SVM produces a model based on a given training data which is then used for predicting the target values of the test data. Given a labelled training set (xi,yi), SVM require the solution of the following optimization problem to perform classification :   '

![](http://imgur.com/w2PBQYJ.jpg)



### Evaluation Metrics
** Prediction Accuracy **  : The accuracy score will be used as an evaluation Metrics in this case. It would be calculated using the number of data points which are classified correctly.  
As it is a classification problem, prediction accuracy turns out to be the best evaluation metrics.  

### Project Design

#### Programming Language and Libraries

- **Python2**  

- **scikit-learn** : Open source machine learning library for python.  

- **numpy ** : Python's numberical library.  

- **matplotlib ** : For plotting graphs and plots.

- **pandas, seaborn ** :  For data reading and visualization.  

#### Operation 
First the dataset would be extracted in a proper format using the scikit-learn functions.    

And then the Support Vector Machine model would be trained using the processed dataset.  SVM functioning is explained above.

Finally we can can calculate the prediction accuracy to see that how well our model is performing.  

This would be the over all functioning of the algorithm.




-----------

