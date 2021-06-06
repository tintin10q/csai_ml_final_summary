---
title: Machine learning summary
date: June 2021
author:
- Quinten Cabo  
numbersections: true
geometry: margin=2cm
lang: "en"
toc-own-page: true
toc: true
---

# 3 types of models
All the ML techniques can be put into another 3 categories.
- Instance based classifiers
    - Uses observations directly without models
    - The only one we have that does this is KNN
- Generative
    - Build a generative statistical model by assuming data comes from a normal distrobution
    - Bayes classifiers
- Discriminative
    - Directly estimate decision rule/boundary
    - Decision trees
---
# Reminder about probability theory 
This is how bayies 
## Variables 
The basis of baiyes probability theory lies with **random variables** that represent events in the future that could happen. These refer to an element whose status is unknown. For example ```A = "it will rain tomorrow"``` 

The **domain** is the set of values a random variable can take, also to be seen as the set all the possible answers to a question, or all the possible events that could happen. This can be the following types:
**Binary**: Will it rain? (yes, no)
**Discrete**: How much will it rain? Integers (With numbers), 
**Continues**: How much more did it rain today? Floating points (With change and )

Then we define the P() function which will give you the chance of a certain event happening. 

P() has these rules:
> 1. 0 <= P(A) <= 1  
> The output is always between 0 and 1.

> 2. P(True) = 1, P(false) = 0  
> true = 1, false = 0

> 3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B) 
> Set Theory. The chance for A AND B is P(A) + P(B) - P(A) ∩ P(B) 
> So A AND B is P(A) + P(B) - (What A and B have in common)  

![Set Theory explanation](./setex.png)

How likely one certain outcome is, is called the **prior**. So the the outcome of a P() is a prior. For example:
- P(Rain tomorrow) = 0.8
- P(No rain tomorrow) = 0.2 
The probability of at least something happening is 1. So the sum of P() for all possible outcomes is one.

So a prior is the degree that you believe a certain outcome before it happens and one is chosen. If you have no other information. 

## Example

|Red| Blue|
|---|:---:|
| 1 |  0  |
| 0 |  1  |    
| 1 |  0  |
| 0 |  0  |
| 1 |  0  |
| 0 |  1  |
| 1 |  1  |

Here P(Red) = 0.5 or 4/8 and P(Blue) = 0.5

## Joint probability
This is where you want to know the prior for A and B or P(Red) and P(Blue) or P(A, B) or P(Red, Blue). It is the chance that multiple events happen together. 
In this case it is = 1/8

Now it is very important if the variables are independent or not. If something is independent then the joint probability is just P(A) * P(B). If it is not independent then you need to talk about conditional probability. 

## Conditional probability
In some cases given knowledge of one or more random variables we can improve our prior belief of another random variable. As soon as you know that variables are not independent you need to talk about conditional probability.  

This is where `|` comes in. (A = 1|B = 1) this asks the question of:
> What is the chance that if B happens A also happens the outcomes where A is 
true if B is true. You could say this as the chance of A given B. 

The joint distribution can be specified in terms of conditional probability.

> You can then combine these two to get the joint probability.
> It looks like this: `P(x,y) = P(x|y)*P(x)`

## Bayes Probability Theorom

It turns out that the joint probability of A and B is the prior of A times the conditional probability of A and B. So `P(a,b) = P(a|b)*P(a)`

Now that is really great because we now we can make the bayes rule:
From joint probability we can say `P(x,y) = P(x|y)p(y) = P(y|x)P(x)` and this gives us conditional probability. 

> It looks like so:
> `p(y|x) = p(x|y)p(y)/p(x)`

Now if we put this back into ML terms. x = features, y = label.
> P(y|x) --> What is the chance for this label y given features x

### Example
![Bayes event table](bay_example.png) ![Bayes event table converted](bayes_filled_in_example.png)

We want to know the prior of passing your exam if your teacher is Larry. We need the following. Let's also replace the P(x) and alike with semantics for this.
> P(x) = P(Larry) = 28/80 = 0.35
> P(y) = P(Yes) = 56/80 = 0.7 
> P(x|y) = P(Larry|Yes) = 19/56 = 0.34 (Prior of your teacher being larry if you passing.)

Ok so with that we can calculate P(y|x) like P(y|x) = `P(y|x) = P(|y)p(x)/p(y)` or `P(Yes|Larry) = P(Larry|Yes)*P(Larry)/P(Yes)`
So if we write it out: `P(Yes|Larry) = 0.34*0.7/0.35 = 0.68` Meaning the final probabilty of passing the exam if your theater is larry is 0.68

# Naive Bayes 
With a the rules from above you can make a machine learning classifier. It is called Naive Bayes classifier. This is a **generative** based classifier. Meaning it builds a generative statistical model based on the data. This classifier is based on the predictions of that model. We do this by just giving the things below we saw before new names.

>- P(y|x): The posterior probability of a class (label) given the data (attribute) -> The probability of the label given the data
>- P(y): The prior probability of a label -> How likely is a certain class?
>- P(x|y): The likelihood of the data given the class -> Prior Probability of the data given a class. Opposite of (y|x)
>- P(x): The prior probability of the data

We are after P(y|x) the label given the data. We can calculate this with P(y|**x**) = P(**x**|y)*P(y)/P(**x**).

In practice, you will always have multiple features so it looks like P(y|**X1** ... **Xn**)

This classifier is called a **Naive** Bayes because it assumes that all the features are independent. This makes it so you P(**X1**|y, **X2**, **X3**...**Xn**) = P(**X1**|y)

This gives us: 

P(y|X1...Xn) = 

![Final formula naive Bayes](final_bayes.png)

His means multiply P(y) with all the priors of your classes and divide by the joint prior of all the data. Then you take the class with the highest **posterior probability**. This is what you choose as your prediction.

> **posterior probability** is the prior * likelyhood or the P(y) * P(**x**|y) of the equation. So we want the highest P(y|x)

### Non discrete data
So far we assumed that the data is always discrete but usually with machine learning this is not the case. The data is often continues. For these types of data we often use a **Gausian model** that assumes your data is normally distributed! In this model we assume that the input X is taken from a normal distribution X = N(mean, sigma). 

## Bayesian Regression
So far we only looked at Bayesian classification. You can also do Bayesian regression. Take linear regression and just put it in the Bayesian model. You base the predictions on probability instead of a single point. You can optimise these with gradient decent even still. and there are multiple best values. You are more after the posterior distribution for the model parameters.  

>**Linear regresssion reminder:**
> 
> y(x) = **W**^T**x** (frequentst view )
>
> Cost (sum of squares): y(W) = 0/2N*Sum(y(x<sub>i</sub>)-y<sub>i</sub>) 
> 
> <u>Predicted - what you found</u>

With bayesian linear regression you formulate linear regression using probability distributions rather than point estimates. This assumes that y was drawn
 from a probability distribution. The sklearn linear regression can actually use baysion regression for regression as well as it is build in.  

![Baysian Regression](baysianregressionformula.png)

So this is just linear regression in Bayesian pretty cool if you ask me its like a whole another way of looking at things.

### Different scikit learn bayes models
There are different implementations of native bayes. The one you want to use depends on the type of your data.
- Gaussian Naive Bayes classifier (FaussianNB())
  - Assumes that features follow a normal distribution 
- Multinomial Naive Bayes 
  - Assumes count data
  - Each feature represents how often something happens. Like how often does a word appear in a sentence.
- Bernoulli Naive Bayes
  - Assumes your feature vectors are binary (Can only take 2 values)
  - Can also be continues values which are precisely split. Like Below 10 is 0 and above 10 is 1.
## Advantages and Disadvantages of Naive Bayes. 
- Advantages:
    - Simple
    - Works well with a small amount of training data
    - The class with the highest probability is considered as the most likely class
    - You get a probability of all the classes
- Disadvantages:
    - You have to estimate parameters of the normal distribution
    - You data needs to be normally distributed
      
      You can use different ones for different types of data.
  
# Decision Trees 
With this technique the idea is build a large logic tree made out of questions about the input data. Each question is a node in the tree. The answer to a question decides which node you should go to next. For example: Is the feature larger than 32? If yes go left if no go right. Going left or right is called **branching** or **traversing** the tree. A node only askes **one question about one feature**. At the end of the tree there are no question anymore and just your label. This node is also called a **leaf**. 

This technique is like having a lot of if/elif/else statements. You get the label by traversing the tree one node at the time by answering boolean questions about a feature in you data. A nice advantage of this is that decision trees are not at all a black box as you basically have the if statements checking your inputs after you made the model. This makes it easier to share the model. 

![Left or right](simple%20tree.png)

> Because the tree is made out of binary questions you don't have to do anything to your data to use it. So you don't have to convert categorical data to something numerical or anything like that as the tree can just directly ask something about a categorical feature. For instance, you can just ask: is the color red? Yes or No. The same thing for numerical values. 
> 
>This is really great as it reduces the preprocessing and it is intuitive.  

Decision trees are a bit like playing guess who:

![Guess who](guesswho.png)

If you go left you might get to a different question as when you would have gone right. Everytime you branch the tree the **depth** of the tree grows. You want the least depth while separating the data as much as possible. A

![More complicated tree](complicated_tree.png)

> All decision boundaries are perpendicular to the feature axes, because at each node a decision is made about one feature only. So if you see strait lines it is probably a decision tree.

The goal behind decision trees is to get the best branching. You get the best branching based on the order you check the features and the thresholds you check for.

You want to split as much "area" as possible like this:

![Each test/node layer in the tree splits your data further. Left is dept 1, Right is depth 2](tree_boundary.png) 

> Every depth increases the amount of decision boundary lines increases with depth as well if that makes sense. This is because every depth down creates exponentially more paths. You are making your decision boundary and eliminating labels as you go along 
> ![Dept 1](tree_depth1.png) ![Dept 2](tree_depth2.png) ![Dept 9](tree_depth9.png) 

#### Example 

In this case what is better X1 or X2?

![X1 or X2](x1-or-x2tree.png)

If you split by asking is X1 true of false and it is true then you immediately get the good y. This is the most error reduction with the least depth. As in that case there are still 0 remaining wrong classified outcomes anymore. Thus splitting by X1 first is better. This will then also be indicated by those three measures.

## Choosing how to split your tree
If you have a tree they using it is easy to use but getting the splits is the tricky part. The only way really is just to try a bunch of values/splits and look at some measures to give indication about the improvement the split gives. Then you pick the split that gives the most improvement. For this we use the decision tree algorithm. This algorithm has to decide:
- What features to ask about
- What values to use in the question (Numbers for continues values, categories for categorical values)
- What order to do the splitting in
- Decide what the outputs are if you get to the leafs

The algorithm works like this:
1. Start from an empty decision tree
2. Split on the next best **attribute**
3. Repeat


## Decision tree classification
Ok but what is the next best attribute? For classification, you can determine it based on these three things:
- Entropy
- Information gain
- Gini index

### Entropy
Entropy is **the level of uncertainty**. The higher the entropy the more uncertainty. 
The formula is:

![Entropy tree](entropytree.png)
You saw entropy in logistic regression. Entropy is the level of uncertainty. It is the probability of your class occurring giving the log of that probability. That means the chance that any instance is that class. You get that by doing: occurrences feature/n. Entropy can never be negative. Instead of P for Bayes we have H for entropy. 

We want to minimize entropy because we want to minimize uncertainty. 

>Probabilities are always between 0 and 1 so the log of the p will be negative that is why this works. As the lower the p the lower the log results.   
![Log of probability](logofp.png)

#### Examples of entropy

![Examples of entropy](exampleofEntropy3.png)

In this case the entropy is the lowest on the right.
You can also calculate entropy for sequences.

![Sequence](informationtheory.png)

Now here are some more things about entropy she discussed. 

What this shows is that if a sequence is more equal then as in there is an equal division of p between classes then the entropy is higher. This makes entropy the method for dealing with unbalanced data.

<u>The idea of using entropy is to try a split and then to calculate what the entropy is after the split. Take the option that has the **lowest entropy** after the split. This option has the least uncertainty.</u>

#### Joint entropy
To find the entropy of a joint distribution you can use this:

![Joint distribution](joint_entropy.png)

She didn't say anything more about it either. So just use this formula.

#### Conditional entropy
Again we use the joint thing + chain rule to get the conditional rule. Conditional entropy  H(X, Y) = H(X|Y)+H(Y) = H(Y|X) + H(X)
If you try to calculate the entropy of a conditional distribution you do it like this. 

![Joint distribution](conditional_entropy.png)

This is asking what is the entropy of X given Y. So instead of likeliness we are after entropy. You can just calculate it with the formula.

#### Dependent vs independent
If X and Y are independent, then X doesn't tell us anything about Y. So H(Y|X) = H(Y). But of course Y tells us everything about Y. H(y|y) = 0.
By knowing X we can decrease uncertainty about Y: H(Y|X) <= H(Y)

### Information gain
With information gain we look for how much information we gain about the features. 
The formula is IG(Y|X) = H(Y) - H(Y|X)

IG(Y|X) is pronounced as information gain in Y due to X. If X is 

If X is completely uninformative about Y then IG(Y|X) = 0
If X is completely uninformative about Y then IG(Y|X) = H(Y) E.G. everything that is uncertain is gone
This is what you use in making the tree. **You want to find the split with the highest information gain.**
> You can see information gain as taking away entropy

### Gini Coefficient
![Gini index](giniIndex.png)

The gini function is called HGini. Gini is cheaper to calculate then entropy because there is entropy has a log operation and Gini does not. This is why this is the default in sklearn. However, Gini only works for binary classification. The idea is the same. Split, fill in the formula, use the split with the lowest gini.

 #### Example:

![Gini example](giniexample.png)

## Thresholds with continues values 
For all of these methods if you have continues values you not only have to try splits but also with different thresholds. Like if X between 10 and 12 or 10 and 13 what gives the best model improvement? 

## Decision Tree Regression
For regression with decision trees you use the **weighted mean square error** to decide on the splits. Try a lot of splits and choose the split that reduces the weighted mean square error the most. 

![Decision tree regression](decisiontreeregresssion.png)

N is the number of training samples at a certain node. y is the true target value y^ is the predicted sample mean. 

As you can see just like with decision tree classification you again get these straight lines in the decision boundary. 

## Complexity of the model 
The tree can get huge quickly. The complexity of a decision tree model is determined by the depth of the tree. **Increasing the depth** of the tree increases the number of decision boundaries and **may lead to overfitting**. For example all these places might have overfitting:

![Overfitting](tree_depth9overfitting.png)

### Ways of reducing model complexity. (hyperparameters)
 There are hyper parameters that limit model complicity. You should set alteast one of these as theoretically you can keep growing the tree forever.
- max_depth = The max depth the tree can grow
- max_leaf_nodes = The maximum leaf nodes that can exist
- min_samples_split = A minimum amount of samples that have to be in a split to make a split.   
There are also more parameters

## Advantages and Disadvantages of decision trees
**Advantages**
- Easy to interpret and make for straightforward visualizations
- The interal workings are capable of being observed and thus makes it possible to easily reproduce work
- Can handle both numerical and categorical data directly without preprocessing
- Performs well on large datasets
- Performs fast in general

**Disadvantages**
- Building decision trees requires algorithms that can find the optimal choice at each node
- Prone to overfitting, especially when the trees' depth increases

# Bias and variance
Before we move on to ensamble learning lets have a reflection moment. 
So far these algorihms were covered:

| Classification      |Regression                     |
|---------------------|-------------------------------|
| Logistic Regresion  | Linear Regression             |
| Linear SVMs         | Linear SVM                    |
| KNN                 | KNN regression                |
| Neural networks     | Polynomial Regression         |
| Kernel SVM          | Decision Trees Regression     |
| Naive Bayes         | Kernel SVM Regression         |
| Decision trees      | Bayesian Linear Regression    |

Some of these are linear, and some of these are not. Linear algorithms create a straight decision boundary line. Non-linear algorithms don't.

These algorithms are known as **weak learners** because they might be sensitive to overfitting. You can overcome this with regularization as we have seen but another way is with ensemble learning. 

The problem that we always have with these models is figuring out how well they will work for unseen data. This problem is called **Estimating Generalization Error**. You can calculate this with: Generalization error = bias^2 + variance + noise

**Error due to Bias:** Bias measures how far off in on average these models' predictions are from the correct value.

 **Error due to Variance:** The variance is how much the predictions for a given point vary between different realizations of the model.

**Noise:** The irreducible error. This is error that you can't do anything about. Noise in the data collection process for example.   

![Bias and variance typically trade off in relation to model complexity](biasvariancetradeoff.png)

Both variance and bias are related to model complexity. If you make your model **less complex** typically you get **less bais but more variance**. If you make your model **more complex** you get **more bais and less variance**. They both contribute to the error, so you want both to be low as possible. Which means you want to find a model complexity that gives a low bais and low variance. It is about finding a balance.   

![Depiction of low and high variance and bias in dartboards](lowhighbaisvariance.png)

This was picture made by scott fortmann roe. [He has a nice further explination about the bias and variance tradeoff.](http://scott.fortmann-roe.com/docs/BiasVariance.html)  

A low bias, and a low variance are the two most fundamental features expected for a model.

Here are more charts that show the effect even more! Made by Duda and Hart.

![Varience and bias even more graphs](biasandvarianceevenmore.png)

The bais and variance are the cause of the underfitting overfitting problem. Because of it you normally expect model performance to behave like this: 

![Bais and variance trade off plotted](baisvariancetradeoff2.0.png)

It seems like the models so far have to deal with this problem. Especially logistic regression, naive bayes, knn, shallow decision trees, linear svm and kernel svm. Some of these have high bias → **low degree of freedom models**. Or they have too much variance to be robust → **high degree of freedom models**. These models do not necessarily perform well by themselves. But who says you can only have 1 model? I don't.

# Ensemble learning
Ensemble methods try to reduce bias and or variance of weak (single) models by combining several of them together to achieve better performance. The ways of doing this are called **Voting**, **Bagging**, **Boosting** and **Stacking**.

## Voting 
Voting is a method were you make multiple models vote on what the output should be. This makes the most sense for classifiers. You just train multiple models and each model votes on what they predict is the correct label. You pick the label with the most votes. This is called **hard voting**. Because the idea of voting is simple it works with any model also neural networks. But we can do better than just counting votes.

Some classifiers also know how "sure" they are of their conviction. For instance naive Bayes. You could give the votes of these models a higher weight.  

Some classifiers will be the most "sure" about a particular part of the space. You could give these models vote more weight for this space. 

If you use weights you in the end you average the results instead of just counting votes. This is called **soft voting**.

More models take longer to train but the results you will get are also based on much more. However, **only combine models if they are not correlated**. You can only do averaging instead of counting if all the models output calibraged/scaled ("good") probabilities.

The sklearn version of this is called VotingClassifier. You give this class a list of other models. 

## Bagging (bootstrap aggregation)
Bagging fits several "independent" models and averages their predictions in order to lower the variance. So this technique is for low bais and high variance models.

Fitting fully independent models requires too much data. This is why we need bootstrapping. **Bootstrapping is taking random samples of the same size (B) from the dataset with repetition.** With repetition means that a datapoint can be in multiple samples. Once you have the bootstrapped samples you can train models on them. These could be all the same or different ones. 

![Bootrapping](bootrapping.png)

So **bootstrapping is creating the samples** from the data and **bagging is fitting models on these samples and taking the average**. You basically create a lot of models based on essentially the same data but you just leave out random data points for every sample. This way each model will have slightly different results. The average of these results will have a lower variance.

![Nice bagging picture](nicebagging.png)

With bagging you can also **approximate the variance or confidence interval of the estimator** by evaluating the variance that all the bootstrapped samples have. 
For regression a simple average is used for classifiation you can use the voting techniques. 

![Approximate the variance process](aproxvariancebagging.png)

Bagging is implemented in sklearn with BaggingClassifier and BaggingRegressor.

### Random forests
Random forests is a bagging method were **deep decision trees**, fitted on bootstrap samples, are combined to produce an output with lower variance. This method is also called a random decision forest. 

Decision Trees can be: 
- **Shallow**. Shallow trees have low depth, these trees have less variance but higher bias, a better choice for sequential methods\boosting
- **Deep**. Deep trees have low bias and high variance. Better choice for bagging method as that is focused on reducing the variance. 

With random forests you do bagging but **you only use (deep) decision trees** to train on the bootstrapped samples.
In addition to that the **set of features you base the splitting decisions on are randomly selected.** So the decision trees only use a subset of the available features to make the splits. This is done to **reduce correlation**. More trees are always beter. 


![Bootstrapping and subset for each split](decisiontreesplit.png)
>So you do double randomization; each tree picks a bootstrap sample and then also only uses a random sample of the features in the picked bootstrap sample to decide on the splits of the decision tree.
> 
Here is the full picture of random forest/random decision forests:

![Random forest full picture](randomforestfullpicture.png)

Random forest is called a strong learner because it is composed of multiple (weak) trees. 

#### Getting results from the forest
How do you combine the results of the trees? You average the tree. The result of the averaging is called a random forest. Like this:

![Random forest averaging](randomforestresult.png)

In this case we averaged 2 decision tree results. In the areas where the trees don't mach the average is 0 (because there are only 2 trees) which means you don't know the answer. If you have more trees you would have places were you are more and less sure instead. To decide on an output for **classification you take the mode** of the classes that were outputted and for **regression you could take the mean** of the values outputted by the trees.  

With sklearn the model is called `ensemble.RandomForestClassifer`. Special hyper parameter that random forests have are:
- `max_features`. Hopefully it is obvious what that does. The recommendation is to pick `n_features^0.5` for classification and n_features for regression. 
- `n_estimators`, This is the amount of trees you want. The more the better. It is recomended to have atleast 100. But the more the better.

## Boosting
With boosting you fit the weak models in sequence unlike bagging which fits in parallel. You do this, so that a model knows about the results of the previous model. This way a model can give more importance to the observations in the dataset that were badly handled by the previous models in the sequence. This way bias can be reduced. So this technique is for a high bais and low variance models. 

This picture shows how you create models sequentially like this:

![Boosting](boosting.png)

The ways discussed creating models like this are **Ada boost** and **Gradient Boost**.

### Ada boost
The idea of adaptive boosting is that you run a weak model in the chain. Then find out which points were **wrongly classified** and then give these points a **higher weight** to make them more important for the next model. This way the next model will try to focus on correctly predicting these specific points. This is called **updating the sample weights**.

The weak model itself also gets a weight based on how well it predicted the data. This is called the **update coefficient** or **the amount of say**. Keep doing this until you get through the chain. At the end you merge all the weak models based on their update coefficients and make the prediction. 

You can use any weak learning model you want but often a decision tree with depth 1 is used. These are called **stumps**.

![Ada boost](adaboost.png)

![Ada boost](adaboost2.png)

[Video going into more depth about ada boosting](https://www.youtube.com/watch?v=LsK-xG1cLYA)

### Gradient boosting
Gradient boosting starts with a simple prediction it could be the mean but it is just a guess. This guess will have a certain error/residuals. Instead of making stumps like adaboost, gradient boost can make bigger trees, but you still set a max size. With these bigger trees it tries to predict the pseudo residuals of the data instead of the target. Then when this tree is made you have to scale the prediction down with a learning rate so the model has less impact on the final result. Then combine this tree with the original prediction, and train a new model based on the new residuals. You should have moved a bit into the right direction from the original prediction. This will give you new pseudo residuals the next tree can try to predict. This tree is then also added to the chain and the residuals should keep getting smaller with every tree you add.

The idea is to find out what the best next tree is every time. I don't really get this one :'(

![img_5.png](img_5.png)

For classification log loss is used and for regression square loss is used. 

![Gradient boosting slide](gradient_boosting_slide.png)

[Video going into more depth about gradient decent its a series](https://www.youtube.com/watch?v=3CC4N4z3GJc)

There is also extreme boosting.

## Stacking 
Trains many models in parallel and combines them by training on a meta model to output a prediction based on different weak models predictions. 

So you choose some weak learners for instance KNN or SVM or a decision tree and then you also choose something as a meta model. Usually a neural network. Then you use the combined output of the weak learners that were trained in parallel as input to the strong learner. 

![Stacking](stacking.png)

You train the weak learners and the meta model on different parts of the data. So you split your data in **two folds**. The first fold is to fit the weak learners and the second fold is for fitting the meta model. Each of the weak learners then tries to make predictions for the observations in the second fold. This gives you the abbility to fit the meta model with the predictions made by the weak learners as input. 
**Doing this does mean you only have half of the data**. But you can use **k-fold cross trainig** (similar to k-fold-cross validatoin) to get around by making sure that all the observations can be used to train the meta model.

![k-fold-training](k-fold-training.png)

In sklearn there is poor man staking and you just use a Voting classifier. 


There is also hold out stacking and naive stacking. 
The idea of this is that with naive you assume that each model in the stack is equally skillful. With hold out you give every model a weight based on their performance in a cross validation. 

![Stacking](holdoutnaive_stacking.png)


### Tree based models 
You often use tree based models as the weak learners in the ensamble learning because the trees can model non-linear relationships. Trees are very interpretable (if they are not too big). Random forest are very robust. Gradient boosting often has the best performance with good tuning. Trees don't care about scaling they can work with any data so no feature engineering. 

We now discussed all the models. There is still more to the story however.

# Model evaluation
How do you beter evaluate your models? Before you awnser that question you should ask what do we actually want to know when we evaluate our model? The answer is the **generalization performance** of the model. We actually don't care really how well the model does on the training data we care about if the training data will be enough to do well on unseen data. We can just get generalization performance of the model by testing it with unseen data. This is only really true for supervised learning because unsupperviced is more qualative.

So far we have just done train/test data. There are few problems with this approach:

- What is the % of test and spit? 
- How do we know if the test data is exceptionally different from the training data?
- How do we know whenther the model is overfitting the data?

The answer to these questions is: We don't know. You have to determine on a case by case basis.  
 
So can we do better? 

## Cross validation

### K-fold Cross Validation (again)

You can also do k-fold cross validation we have seen this already in the other summary. The idea is making multiple models of your data with different parts being part of training and testing every time. k is the amount of models that you want to make. 

#### Benefits of k-fold cv

- Leaves less to luck, if you get really good or bad performance by luck then this will show in the results. One of the results will be an outlier. This can happen a lot with imbalanced data. 
- Shows how sensitive the model is to the training data set. High variance between fold scores means high sensitivity to the training data. 

#### Disadvantages of k-fold cv

- Increases computational time because you train more models
- Simple cross validation can result in class imbalance between training and test. This would lead to lower scores than you could really get. To get around this we can do **stratified cross validation**. This is where you make sure that there is no class imbalance in the different folds. 

You can also set k to N and then you make the whole model with all the data and only test with one point. This as you know is called **leave one out validation**. This is very time consuming but this can be the **best method for small datasets** as it generates predictions on the maximum available data. Small datasets also decrease the training time again. This can also be useful to find out which items are regular and irregular from the point of view of the dataset. 

### Shuffle split  
The idea of this is that you do k-fold cross validation but which fold you use is picked at random. This:

- Controls test-siz, training-size and number of iterations. You can again also do **stratified shuffle split cross validation** what a mouthfull. 

![Shuffle Split](shuffle_split.png)

### Cross validation with groups

In cases that groups in the data are relevant to the learning problem you have to make sure that you **keep the whole group either in the test set or training set**. For instance with the machine learning take home we saw that some faces were in the dataset multiple times but with different expressions. In these cases it is important to keep the whole group of faces in either the test set or training set otherwise you get better or worse performance then is real. 
Another example is with **time data** this is important to keep in order. 

## Tuning
Tuning is where you improve the models generalization performance by **tuning**\adjusting the hyperparameters. You can do this with grid search. Grid search is just a fancy word for training and testing with a lot of combinations of hyperparameters and using the best. You however should make clear before you do grid search which values you want to try. 
**Simple grid search** is when you just try all the possible combinations of the hyperparameters you specified. 
**Grid search with cross validation** use cross validation to evaluate the performance of each combination of parameter values. 

The **problem with grid search** is that if you optimize the model based on the test data that the **test data is not independent anymore!** To fix this requires another final test set that you use with the best model you found with grid search. This is where the **validation set** comes in. You use the validation set to get a final score on the best model.

When doing grid search you can save all the results you got for the different models and create a grid. This grid can be very usefull as ussually things that are close on the grid have similar performance.  

![Grid search grid](grid_search_grid.png)

## Metrics for classification

For classification, you can get a couple of extra metrics besides training testing and validation score. These are:

- **Accuracy:** How many data points were correctly classified? 
- **Precision:** When a classification for a class, is it a true positive?
- **Recall:** Of a certain class how many are correctly classified. Also called sensitivity. 
- **F1 score:** A way of combining precision and recall. Defined as the harmonic mean of precision and recall.  

How to calculate them:

![Metrics for classification models](metricsforclassification.png)

### Binary classification

If you only use accuracy you can get some problems if you have imbalanced data. This is because you might get class A right all the time but class B only sometimes but if most of the data is class A you will still have a high accuracy. 

![Accuracy can be a problem](problemwithaccuracy.png)

What scores you attach to most value to depends on the **goal setting**. Lets say you have a pacemaker factory then you have to be sure that every pacemaker is 100% save, and you are ok with trowing away some pacemakers. Even some false negatives. Because economically speaking a false positive (working but looks broken) will cost you 10 euros while a false negative (broken but looks ok) will cost you a potential human live (wich will cost you more than 10 euros).

Changing the threshold that is used to make a classification decision in a model is a way to adjust the trade-off of precision and recall for a given classifier. This gives you the precision recall curve. This is useful when you make a new model. 

![Precision and recall curve](recal-precision-curve.png)

So lowering recall gives you more precision but at some point recall drops off really quick. In this case a **precision zero** gives a good balance between precision and recall. These curves are better for imbalanced data then Roc curves. 

Another way to look at this is the Reciever operating characteristics curve (Roc). Instead of reporting precision and recall it shows the false positive rate (FPR) against the tue positive rate (TPR). Here is how you calculate those: 

![FPR and TPR formulas](roc-curve-formula.png)

Then using that you can make the roc curve it looks like this:

![Roc-curve](roc-curve.png)

**The more area under the curve the better**. If you had complete random guesses the lines would go directly through the middle. This model is better and there is a lot of area under the curve. More better for balanced data. 

### Multiclass classification metrics 

How do you use these extra metrics for classification were you have multiple classes? You make a **confusion matrix** were you calculate the metrics for every class.

They look like this:

![Confusion matrix](confusionmatrix.png)

This is the takehome assigment where the first collum is for image type 1 second collum for type 2 and thirth model is type 3. 

This also gives you differnt types of F1 score options: 
**Maco-averaged F1:** Average F1 scores over classes. So here you assume that all classes are equally important. 
**Weighted F1** Mean of the per class f-scores weighted by their support
**Micro-averaged F1** Make one binary confusion matrix over all classes then compute recall, precision once (all samples are equally important)

You can calculate all the metrics its about what metric you attribute the most meaning towards. 

## Metrics for regression models. 

- **R2**: Easy to understand scale
- **MSE**: Mean square error, easy to relate to input
- **Mean or median absolute error**: More robust metrics meaning they are less influenced by the distrobution characteristics of the data. 
- **Negative version**: Negative versions of the other metrics. Something that decreases if the model gets better. 

You can also plot prediction or residual plots with regression models. 

![Regression metrics plots](regression_model_metrics_plots.png)

## Dealing with unbalanced data

Imbalanced data is when you don't have the same amount of data per class/outcome.
This can happen due to:
- **Asymmetric cost** some labels being more important that others. 
- **Asymmetric data** the distribution of your labels in the dataset is unbalanced.

Most datasets follow a Zipfian distribution. The amount of the next class is about half of the current class. 
How to deal with it:
- Change the data by adding or removing samples. This is called under and over sampling. 
- Change the training procedure.
- Don't deal with it.

With **random undersampeling** you lose data but this also makes training faster. With **random oversampling** you get more data so more chance at overfitting and slower training but at least you don't loose data. 

You can also do an **edited nearest neighbors method**. The idea is to remove all the samples that are misclassified by a KNN from the training data. Cleans up the training data by removing outliers. 
You can also do **condensed nearest neighbors method**. This iterativly adds points to the data that are misclassified by KNN and tries to do the oposite of edited. This way you get a dataset that focuses on the boundaries and so usually removes a lot of points. This is not really used a lot. 

![Edited vs Condensed KNN](edited-condensed-knn.png)

![Summary of video 8](summaryv8.png)

Again the idea is to calculate multiple metrics and focus more on the ones that are important to your goals and needs.  


# Preprocessing and Feature Engineering

When you trow your data at a machine learning model as is you might not get optimal performance. You can do things to your data that do not change the sementics of your data while improving the generalization ability of your model.

## Scaling
We have seen this already a lot. The idea is the you scale the data to be on the scale. This makes it easier to compare data that is on scales with small and big numbers that is not so easy to convert. For instance km and hrtz. 

### Standart Scaler
With this method you calculate the z score for every data point. Effectively you calculate how far something is from the mean.  

![z-score](z-score.png)

This works well for non skewed data. 

### Robuts Scaler
The same formula but instead you assume a normal distrobution and you use the median instead of the mean and you use the interquartile range instead of the standart deviation. This is better for skewed data and deals better with outliers.  

### Min Max Scaler
Shifts data to the 0-1 interfall. Take the maximum value of your dataset and mimimum value of your dataset and just calculate where the other data points lie in between those and then you can get a 0-1 scale. Nice. 

![Min max scaling](min-max-scaling.png)

### Normalizer
ojas;doilkajfthis method does not work for one feature but for aklopw;'/ojas;doilkajfgklo ,./nvbhll the features of a point. It does this by seeing all the features of a datapoint as a row or vector. The idea is then that you scale all the data of a point (one row) so that the norm (of the vector) becomes 1. Then you divide each point of the row by the norm. This is method is not used that often. Mostly used when the direction of the data matters and it could be helpfull for histograms. 

> The norm of a vector is the square root of the squared elements of the vector. 

### All techniques in one graph:

![Scaling techniques](data-scaling-techniques.png)

## Transforming the data
Instead of just scaling the data you can also transform it. There are many techniques for this. Not only for machine learning also for stats in general. Most models perform best with normally distributed gausian data. Not every model for instance KNN does not most models benefit from it. 

There are even methods to find the best method to transform your data to a Gaussian distribution. For instance Box-Cox and Yeo-Johnson transform. Both are power transform methods. The difference between these 2 methods is that Yeo-Johnson can also do negative numbers but therefore is also less interpretable. **With these techniques you can automatically estimate the parameters so that skewness is minimized and variance is stabilized.** Here is a visualization of the different techniques: 

![Univariate transformations](univeraiate-transformations.png)

These methods are really great because you can just almost blindly transform your data to a normal distribution. You could choose your own parameters to make your data more interpretable. For instance in this case a transformation to the log scale is apparently better: 

![Log transformation](log-transformation.png)

## Bining
Binning (also known as discretization) is a preprocessing method to make linear models more powerful on continuous data. The idea is to separate feature values into n categories that are equally spaced over the range of values. You do this by separating the features into n categories by a single value (usually the mean) and then you **replace all values within a category by a single value**.
This is effective for models with only a few parameters such as regression, but it is not effective for models with a lot of parameters like decision trees as can be seen in the picture below. 

![Binning in action with linear regression and a decision tree](binning.png)

## Missing values imputation
Missing values imputation is about dealing with missing values in the dataset. With real data there are various reasons why data would be missing.  Missing values might be represented in different ways common missing value representations are **None, blank, ~~0~~, NA, Na, NaN, Null, undefined ......** You should **NOT** use numbers to indicate missing data as this might not be picked up on. Always use strings or something else inconvienient. You could just trow away the data that contains missing values, but we can do better than that. You want to do this because it is sad to throw away a whole vector of data only because one scalar is missing. 

This is how it would look like:

![Missing values](missing-values-imputation.png)

We can make a model to impute (predict) the missing values. Different techniques for this include:
- **Mean / median** replace missing values with median or mean of the data. Not a great technique not very flexiable. 
- **KNN** use the average of k neighbors feature values. This is much more flexible. The mean is basicaly k = n. 
- **Model driven** using another model to impute the values like random forest. 
- **Iterative** using a regression model with all features expect one to predict the missing features and then do this for all the missing features. You can use the recited values for the next values.

![Data imputation](data-imputation.png)

## Dealing with categorical data
Data is often not nice and numeric but categorical. Child/adult boy/girl/other True/False. Data can be categorical - ordinal - interval - ratio. We have seen this before there are ways to deal with this. Usually by converting the categorical data to a number. Remember you don't have to do this for decision trees. To change the categorical data into numbers you can use **one hot encoding** (making multiple boolean features for each possible category), or **a count based encoding** using an aggregation (how much something occurs). Good for high cardinality (a lot of possible classes). We saw this in other courses already.

## Feature selection
Feature selection is choosing what features you want to use when training your models. You do this to:
- Avoid overfitting
- Lower compute time
- Less storage for model and dataset

There are 3 different strategies discussed. 

### Univariate statistics 
With this feature selection technique you look at each feature individually and **remove features** that do not have a **significant relationship** to the target. You can either say keep 20 features with the highest confidence, or you could say keep the features with a confidence value higher than a threshold. How to get confidence is related to the ANOVA method. The idea of this technique is to keep features that have a high statistical significance to the target. You can use f-value or **mutual information** (not linear). Or you can calculate both. 

### Model based selection
This is for getting the best fit for a particular model. Ideally you do an exhaustive search over all possible combinations but this is not feasible as it would take too long. Thats why you use heuristics. But this is lasso regression or linear models, tree based models. You can also do **multivariate models** those are linear models that assume linear relation. The idea of this technique is to just try a lot of combinations of features and find out based on the results and heuristics which combination works the best. 

### Iterative approach
Start with the most important feature and keep adding until you have the amount of features that you want. This is used more for statistical models. You can also do this backwards were you drop features the least important features until you have the amount you need.
This method is quite computationally expensive as every time you try leaving out a feature you have to train again. 

You can rank the features using RFE -> Recursive feature elimination and selection. This is doing the iterative approach backwards and forwards to come to the best combination of features. 

## Dealing with Text data
Most machine learning models really like their data to be numerical but when you get text data this gets more difficult and preprocessing steps get more involved. How do you represent Text data in a matrix format? Text data has no predefined features. The most common way to deal with this problem is to use **Bag of words**. 

![The bag of words technique steps](bagofwords.png)

Before you do anything you need to **tokenize** your raw text. This means turning the raw text into a list of the raw elements of the text. Tokenizing is a whole subject in its own right. Are . , ; " < > ' tokens for instance? You could do **stemming** where you reduce words to their root like walking to walk. You could do **lemmatization** were you replace words with words from language databases. You could **restrict the dataset** by removing words that only occur once to reduce matrix size. You could remove words with low semantic content like the, is, a. Or you could not do any of these things. Do you split the words as 1 word or do you try parts of sentences? What about misspelled words? All dials you can tune in the tokenizing step. Tokenizing is easy to do but hard to do well. 

Once you have tokenized your all your data you have built a vocabulary with the tokens. Then the next step is to create a sparse matrix encoding out of this vocabulary. Where for every token you say if it appears in a certain document or not. You could also use count instead of boolean to represent how often a word appears in a document. 

> A document is a very general term that can mean many things. It could mean a paragraph, different chapters, different files, different sentences etc.

Instead of boolean or count you can also use the **TF-IDF** value of a token. This will tell you how informative a certain token is for a particular document. 

Term Frequency (TF) = Number of times a token appears in a document
Inverse Document Frequency (IDF) = log(N/n). Where, N is the number of documents and n is the number of documents the token has appeared in. Rare tokens have high IDF and frequent words have low IDF. Thus, this highlight words that are distinct. This kind of leans into information theory where the things that are less common actually give you the most information. 
You calculate TF-IDF as TF*IDF.

![TF-IDF example](TF-IDF.png)

As you see the word day has a higher score then beautiful.
