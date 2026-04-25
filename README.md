KNN
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

Kmeans:
In this random datapoints are there and we choose k points say k=3 , in this =we calculate the distance of each point from these 3 points and group them together based on the shortest distances, for example if one point is closer to 2 than 1 or 3 it will be classified as 2 and then it will be grouped with it after all of it is done the the position of 1,2,3 changes as we take the average distance in each cluster. We repeat the same process again and again until there is no significant change in the position of the centroids and distance clusters are formed
These 1,2,3 are called centroids which are randomly chosen.
To select the right number of k we use the elbow method the plot is between WCSS(Within Cluster Sum of Squares) and k values
Distortion:
Means how well the clusters represents data, low distortion better clustering
￼Inertia:
Calculates same thing as distortion but is sum of squared distances

￼
SVM:
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
🔵 🔵 🔵
🔴    🔴
🔴 🔵 🔴
🔴    🔴
🔵 🔵 🔵

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

Distance to 🔴₁ → K = 0.9 (very close!)
Distance to 🔴₂ → K = 0.4
Distance to 🔵₁ → K = 0.1 (far away)
Distance to 🔵₂ → K = 0.05

Red influence = 0.9 + 0.4 = 1.3
Blue influence = 0.1 + 0.05 = 0.15

    → Classified as 🔴

—> sigmoid  kernel : is more neural related


Decision trees:
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


DBSCAN:
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
