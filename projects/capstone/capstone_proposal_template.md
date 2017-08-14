# Machine Learning Engineer Nanodegree
## Capstone Proposal
Charu Chhimpa  
August 14th, 2017

### Domain Background
Advancement in the field of medicine have greatly improved our quality of life which can be clearly seen from the life expectancy rate. From 1816’s stethoscope to today’s handheld ultrasound machines, doctors have steadily adopted technology to advance healthcare. In past decades the healthcare community has taken major steps by adopting electronic healthcare measures. Machine Learning can contribute significantly in the field of healthcare, it has great usage in cases like early disease detection, finding signs of early breakouts of epidemics, using clustering to figure out regions of epidemics, or finding the best air quality zones in countries with high air pollution.  

Heart disease is the leading cause of death in the world over the past 10 years. Researches have been using several data mining techniques in the diagnosis of heart disease. Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin, or when the body cannot effectively use the insulin it produces. Most of these systems have successfully employed Machine learning methods such as Naïve Bayes and Support Vector Machines for the classification purpose. Support vector machines are a modern technique in the field of machine learning and have been successfully used in different fields of application. 




### Problem Statement

In this project we will train a model using Support Vector Machines to predict that a human being is suffering from a  heart disease. We will consider several features to do this classification.   

** Task : **  To predict that a person is suffering from a heart disease or not.  

** Performance : ** Accuracy - No. of correct predictions.  



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
We will 

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

