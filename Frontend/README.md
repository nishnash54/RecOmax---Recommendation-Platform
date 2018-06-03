# RecOmax
Recommendation Platform


### The vision
In today's world, data analysis coupled with the power of Machine learning and Artificial intelligence (deep learning) is helping companies solve the most complex of problems. We designed RecOmax as a ready to use platform that will help P&G predict sales of a specific item in a specific store based on historical sales data and complex trend analysis. We aim to build end to end solutions that benefit the client and provide them an edge over their competitors.


### The build
The build can be divided into 3 main sections. These are the Recommendation engine, the Prediction engine and the Client facing data dashboard (report). These fields are elaborated on below

* Prediction engine
This engine works off a Kaggle data set. The main aim of the engine to is predict the sales of a specific item in a specific store location for the next month. The engine makes two types of predictions
    - When historical sales data is present, that is when we have sales data of a particular item from particular shop and
    - When historical sales data is absent, the engine analyses trends in item sales based on factors such as store locations, date-time features and various other measures.
The prediction engine has a Root mean square error of 1.33.

* Recommendation engine
This engine works on the provided data set. It aims to recommend to the end user similar items based on the current selection. The data set provided looks minimal, but we used data analysis to generate features for each individual item from the given data. As this is an online data set, the trend analysis of this data is integrated into the Prediction engine also.

* Data dashboard
A fully interactive dashboard to present our final platform to the client with data visualizations and information regarding the working of various features in the platform. We used the professional Business Intelligence tool Power BI to prepare the client end report (dashboard)

### Project evolution
Overtime we have planned on making this project in to a full fledged tools with big data integration and real time online data integration. Using big data tool such as Hadoop and Mapreduce, we want to take this project one notch higher by computing tremendous amounts of data. Another plan is to based on the fact that online sales data is available almost immediately. This means that using that data will give the engine an edge over any older historical data prediction model.