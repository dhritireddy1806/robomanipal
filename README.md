**KNN**:
It looks at the k closest data points and makes prediction based on majority class or their average values
In this we choose a value k and its chosen carefully so that it doesnt become too big . If the dataset has a lot of outliers a biog value of k should be good but if its too big the model becomes too simple and it wont  catch simple patterns and will underfit.
There are multiple ways to choose k:
—> cross validation : data divided into k parts , model is trained on some of these parts and tested out on remaining parts
Example :there are 100 samples and k=5 this is what it looks like:

Fold 1: [TEST ] [train] [train] [train] [train]
Fold 2: [train] [TEST ] [train] [train] [train]
Fold 3: [train] [train] [TEST ] [train] [train]
Fold 4: [train] [train] [train] [TEST ] [train]
Fold 5: [train] [train] [train] [train] [TEST ]

Then we take average of all 5 test results and then use that k relevant to that score	

—> elbow method: an error/ accuracy vs k values graph is plotted and the point where the curve changes direction to look like an elbow we use that k value

Both these methods work together to give the best k value . Former one tell us the score for each value of k and the latter tells us the to stop at the bend when we plot all the scores

—> odd values for k : this helps avoiding ties when deciding which class is the most common among the neighbours
To find out which neighbours to use we use some methods:

—> Minkowski distance: this is the family of distance and it has 2 of the bottom ones as its family
If p is 2 then its euclidean and if p =1 then its Manhattan
—> euclidean distance 
—> Manhattan distance 

**Kmeans**:
In this random datapoints are there and we choose k points say k=3 , in this =we calculate the distance of each point from these 3 points and group them together based on the shortest distances, for example if one point is closer to 2 than 1 or 3 it will be classified as 2 and then it will be grouped with it after all of it is done the the position of 1,2,3 changes as we take the average distance in each cluster. We repeat the same process again and again until there is no significant change in the position of the centroids and distance clusters are formed
These 1,2,3 are called centroids which are randomly chosen.
To select the right number of k we use the elbow method the plot is between WCSS(Within Cluster Sum of Squares) and k values
Distortion:
Means how well the clusters represents data, low distortion better clustering
￼Inertia:
Calculates same thing as distortion but is sum of squared distances

￼
**SVM**:
The shortest distance between the threshold and the observation is called the margin

Maximum marginal Classification : find the hyperplane and separates the data with the largest possible margin

When we allow misclassifications the distance between the threshold and the observation is called soft margin
If we are using soft margin to locate the threshold	 then that is basically using  Support Vector Classifier. This is basically Maximal margin classifier with tolerance

When data is 1D the the SVC is just a dot

When data is 2D then the SVC forms a line and has soft margins on both side one lying inside soft margins are misclassified and the ones lying outside the margin are not classified

When data is 3D then the SVC forms a plane and we classify data by determine which side of the plane they are on

When data is in 4D then SVC is a hyperplane

SVC cannot work in data in which wherever we put the classifier, it misclassifies ,thats why we use SVM

Main idea behind svm is :
Find data in a relatively lower dimension (mostly 1D)
Then move this data to a higher dimension
Then fins support vector classifier to classify the data into 2 groups

Svm uses kernels to make a margin . In some data like 
*b *b *b
*r    *r
*r *b *r
*r    *r
*b *b *b
Its hard to draw a fair margin  so the kernel lifts these datapoints from a lower dimension to a higher dimension so that its easily separable
But transferring every datapoint to higher dimension is computational expensive so kernel computes they  dot product of the data in high dimension without even going there

Types of Kernels:
—> linear is basically used normally when there is no dimension increase and the data is linearly separable.

—>There is a type of kernel called the polynomial kernel which increases the the dimension be setting d , degree of polynomial, its transform lower d data into higher d data by adding new feature which are the combination or modification of the existing features
￼
A and b are 2 points , r is a bias constant and d is degree

—> radial kernel uses infinite dimensions unlike polynomial  one because of which it has infinite flexibility to create a boundary in any shape . It basically classifies a new point based on how far it is to the training point, closer the point more is the influence. It also has a gamma factor , higher the gamma tighter is the boundary which means each point influences only its neighbour.
￼
Closer to 1—> more similar
Closer to 0—>more different
New point ?

Distance to *r1 → K = 0.9 (very close!)
Distance to *r2 → K = 0.4
Distance to *b1 → K = 0.1 (far away)
Distance to *b2 → K = 0.05

Red influence = 0.9 + 0.4 = 1.3
Blue influence = 0.1 + 0.05 = 0.15

    → Classified as *r

—> sigmoid  kernel : is more neural related


**Decision trees**:
When it predicts numbers —> regression tree
When it classifies into categories—>classification tree

Root node—>Branches—>Leaf nodes

For each part we calculate mini impurity which tells guys how mixed up the group is 
Its basically	1-(probability of yes)^2 - (probability of no)^2
The less the mini impurity more is the chances of that column to be chosen to be in rhetorical decision tree
Impurity is always between 0 and 1
We draw adopted line to show the distance between the observed and the predicted value and this line is called residual and they are used to quantify the quality of the prediction
We choose the threshold with the lowest sum of squared residuals
We calculate the sum of square residuals at each step	
The lower sum of squared residuals become the root of the tree

For more than one predictor we find the optimal threshold for each one and pick the one with the lowest sum of square residuals to be the root of the tree
When we have some fewer	than minimum observations in a node that node becomes mes a leaf else we keep repeating the process until we can no longer split the observations

We do pruning also which means cutting off unnecessary branches from the tree
The whole idea is preventing the overfitting of training data so it can perform better in the test data

Types of pruning
—> cost complexity pruning:
We find the sir for each tree with reducing leaves and then we find the tree score. The one with the lowest tree score we choose that treee to be more appt
Tree score= ssr+alpha T
Alpha is tuning parameter found using cross validation
T is the no. of leaves

Step by step procedure 
Using the data build the full size tree
(Put alpha =0 to remind that this tree has the lowest score.We slowly.increase alpha until pruning leaves give us a lower tree score, and repeat the process until we cannot remove leaves anymore)
Take the dataset and divide it into train and test
And just using train data use alpha values to bid a full size tree and find the tree score for different values of alpha and find the one with the lowest tree score
Now calculate ssr using testing data
Create new training and testing data
Build new trees using alpha we found before using terrain data
Then we get sir using new testing data
Keep repeating until 10 fold cross validation is done and find average of the alpha values
Then we go back to original trees and find the tree which corresponds to the final alpha value

it has low bias and high variance


**DBSCAN**:
Its identifies clusters in higher dimensions even if they are nested
The procedure is
There are some points which are close to each other and there are some points which are far enough to be called non core points .
We take a random point as a core point from the nearby grouped data   points.
Then we take in all the points near to that core point as a part of that cluster and the each point which was added acts a s a core point and takes in other points as core points of that cluster.
Suppose in the vicinity of a core point there are both a core and a non core point , we first take in all the core points and later come back to the non core points
After finishing all the core points we take the core points near to the non core ones and take them in
Once we take the non core points in we cannot use them to take in other non core points.
This forms one cluster and similarly other cluster is also formed
Suppose a point is close to both the clusters then when are making the first cluster we take that point in into the first cluster 
The other points which are not in either clusters are then labelled as outliers.


**Naive Bayes**:

In this we decide where a particular datapoint belongs to among the present groups 
Like for example it is used to filter spam messages 
We make initial guess with the probability which is called prior probability	
We take few words called friend dear lunch money and find their probabilities in both normal and spam messages
Now when we take a sentence like dear friend we find the probability of it occurring in normal and spam . Suppose it has a higher probability in normal messages , it is then classified as normal messages
P1=P(N)* P(DEAR)*P(FRIEND)
P2=P(S)*P(DEAR)*P(FRIEND)

But if we take a line like lunch money money money , its looks like a spam message 
P1=P(N)* P(LUNCH)*P(MONEY)^3
P2=P(S)*P(LUNCH)*P(MONEY)^3=0.  as P(LUNCH)=0

Suppose lunch doesnt occur anywhere in the spam messages so the probability off the entire sentence to occur  in spam is zero and it is classified as a normal message which is wrong so we add a dummy count and find the probability . No lunch doesnt have a zero probability so this message gets rightfully classified into spam folder

The word naive is used because it ignore the dependency of features on each other and treats them as independent features
Like it doesnt care if its Dear Friend or Friend Dear it has the same liklihood

Now there is a difference between probability and likelihood 
Former has fixed distribution like mean or std deviation and we try to predict the outcome by taking the area under the graph
Latter haș fixed data and we try to find the mean or std deviation which suits the data the most by taking the corresponding y coordinate for the x value

This naive Bayes is called multinomial naive Bayes
There is another variety called the gaussian naive Bayes which uses the gaussian distribution curve and the ln function to find the likelihood of a value.
It uses ln to prevent underflow which means the computer can only take small number to the extent it can track after that it will show the wrong answer

Naive Bayes has low variance(because it is stable as in it the features dont really depend on each other so it doesnt overfit)  and high bias(because it make a strong decision and misses the dependencies of various feature)


**Ransac (Random sample consensus)**

It is used to make outliers into inliers. Its a trial and error approach

Sample: pick s a few data points and assumes they are inliers and make a line pass thru them
Compute model parameters using sampled datapoints by using those datapoints as the standard inliers meaning find the datapoints which lie in the vicinity or on the line
Score: take the datapoints which I haven’t used so far and score them how consistent they are with the model computed
Repeat this again again and final model should have the highest number of inliers
We need those many no. of trials so that with probability p atleast one random sample is free of outliers
One inlier = 1-e. Where e. Is the outlier ratio

T= log(1-p)/ log(1-(1-e)^s).   s= random no. of samples
Small s is better when there are a large no. of outliers


**Random forest**

In this we make many decision trees and find the the classification more accurately
The steps are:
First make a bootstrapped data I.e. randomly selecting samples from the original dataset. it allows duplicated samples as well and some samples are left out which are called out of  boot samples
Then we choose 2 values as contestants for the root node and the one which is chosen separates the samples very well 
The we choose other 2 values and choose one of them for the branches and continue till we all the branches are over
Like this we make many different trees with different bootstrap data
Then we choose a sample for prediction and we run it through all the trees in the random forest and it get classified into the class which has more probability 
To test the accuracy of the forest we use the out of bag samples as well to predict if the forest gives the right answer or not if majority	 is  yes then its working fine
Bootstrapping the data and using the mean to make a decision is called bagging
The proportion of out of bag samples which were incorrectly	 classified show an out of bag error

To get the most relevant forest we 
Build a random forest 
Estimate the accuracy of the forest 
Go back and change the no. of variables er step used at a time to make the forest
And redo everything

For filling missing values:
To keep track of similar samples we use a proximity matrix after filling in the missing values in the dataset
After this we find weight frequency for both the class
Which is wt freq= P(YES)*weight for yes
The one class having higher wt freq is use for classification
Weighted avg= sum(sample’;s wt* samples’s weighted avg wt) thats what is filled in the missing value 
Repeat 6-07 times until missing values converge

**XGBoost**
For regression trees:
In this we use different values of threshold(average of two observations) , find the similarity score and calculate the gain for each leaf to determine how to split the data then we prune by calculating gain - gamma :
Negative—>remove branch
Positive —> keep it
Then we calc output values for the remaining leaves
When lambda is greater than 1 its easier to prune trees as it shrinks the similarity scores and gain will be< than gamma
If lambda is 1 it will prevent overfitting of data	
Eta is the learning rate in xgboost by default is 0.3
We keep taking the residuals and keep building trees until the residuals keep getting smaller
First we get the prediction which is 0.5 by default and find the residuals each tree starts off as an single leaf where all th residuals go 
Then we calculate similarity score using lambda
Then we take 2 observation take their mean and take it as the threshold and calculate similarity and gain then move the threshold again ad repeat until the threshold can no longer be changed then we prune

For classification trees:
The same initial steps as reg
We find cover which helps us to determine the min. Number of residuals in each leaf . Its basically the denominator of similarity score -lambda
We take the initial prediction and then find log(odds)=log(p/1-p)
And then find the probability
After we get the new residuals (output values) we build the new tree fitting these residuals
New thing is the previous probability is not the same or all residuals
And we keep on continuing this until. We reach a mx number of narrowed residuals
We calculate output value after pruning

**PCA**
its is done for unsupervised data to reduce the dimension of dataset from higer to lower
we need to reduce 
--> dimentionality curse: more the dimensions less the accuracy
--> to improve model performance
--> to visualize data
Feature selection:
we found the covariance
if its positive x and y both increase and vice versa
if its negative x decreases and y increases
if its 0 x and y have no relation
pearson correlation=cov(x,y)/(σx*σy) its between -1 to +1
more towards +1 more positive correlation between x and y
more towards -1 more negative correlation between x and y
more towards no correlation between x and y
it needs maximum variance between classes
The steps are :
--> standardize data
--> covariance matrix of x and y
--> find out eigen value and vectors
-->you will get some eigen values the bigger value is pc1 and the second bigger value is pc2
pc1 will be capturing maximum variance
basically you transform the present axis and make new axis so more variance is capturedand the spread is less

**LDA**
has the same pirpose as PCA but works on supervised data
it minimizes the variance within class and maximizes the variance between classes
steps:
-->compute class mean for all classes
--> derive Scatter matrix (unscaled covariance matrix ) for all classes 
-->compute within class scatter matrix Sw=S1+S2
-->compute between class scatter matrix Sb
-->compute eigen values and vextors
Sw^-1*Sb w=λ w
we will get eigen values as λ1,λ2 if its 2D
--> sort eigen values and take top k values
--> find eigen vectors corrwsponding to top k eigen values
--> obtain the LDA by taking dot product of eigen vectors and original vector


**T-SNE**
--> determine similarity score of all points 
a)first determine the distance d between selected point and other points 
b)plot them on the normal distribution curve the line joining the curve and the point is called unscaled similarity. like this do for all points
c)scale the unscaled unsimilarities
                  scaled score=score/sum of all scores
  scaling is required to make sure similarity scores of tighter and relatively loose clusters are equal                
-->project the data on the 1D line  then measue the distance between the target and all points and plot on the t distribution cureve
--> repeat the previous process till we get the heatmap or graph similar to the original cluster
T-SNE moves the points little by little and is used to prevent clusters from clumping in the middle
