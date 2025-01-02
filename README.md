# "Credit Card Approvals: SVM vs. k-NN Classification"

#### Introduction

This project is an analysis of the Credit Approval Data Set from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Credit+Approval). The data contains anonymous credit card applications of 654 data points, 6 continuous and 4 binary response variables. The last column indicates whether the application was positive or negative. 

The goal is to build a predictive classification model that will determine if a credit card application will be approved or not based on various predictors. The first part of the analysis will use a Support Vector Machine (SVM) classifier by implementing the `ksvm` function in the R `kernlab` package. This function will produce a classification equation that can be used to predict whether or not a credit card application will be approved. We will then evaluate the performance of this classifier equation using the entire data set. 

We will repeat the process using another method called the k-Nearest Neighbor (k-NN) algorithm using the function `kknn` contained in the R `kknn` package. We will then compare its performance to the SVM model. 

#### Support Vector Machine (SVM) Analysis

First, we read the our data set and setup the SVM by calling the `ksvm` function:

```{r message=FALSE}
library(readr)
library(kernlab)

# Read the dataset file
data <- read_delim("C:/Users/leegu/OneDrive/Desktop/Projects/SVM/data 2.2/credit_card_data-headers.txt", delim = "\t")

# Call ksvm 
model <- ksvm(as.matrix(data[,1:10]), data[,11], type="C-svc", kernel="vanilladot", C = 0.00141, scaled = TRUE) 

model
```
The `data[,1:10]` calls the first 10 columns of our dataframe, which represents the predictors. These columns are converted into matrix format because the `ksvm` function requires the data to be in matrix form. Our response variable is represented by `data[,11]`, which is the the 11th column of the data frame. `C-svc` is the type of support vector machine model, where "C" is a parameter that controls the margin of our SVM. 

C determines the severity of the margin violations that the model will tolerate. The higher the C value, the more narrow the classifier margin becomes. A narrow margin will tolerate less violations in classification. On the contrary, a lower C will have a wider margin that tolerates more misclassification. In our case, the value of C is 0.00141. This value was computed by manually testing different values of C. In general, it was found that smaller values of C produced higher accuracy results (this was tested in the section titled "Evaluating the Performance of the Classifier." ). From an accuracy standpoint, we can deduce that the ideal classifier margin of our model should be wide. 

Vanilladot is simply a type of kernel, which allows the data points to be linearly separated. Finally, we scale the data in order to address the sensitive nature of SVM models when it comes to the range of values we put in. Setting `scaled = TRUE` will ensure that all predictor values will contribute equally to the SVM, regardless of how they are scaled. 


#### Getting the Equation for the Classifier
Next, we calculate the coefficients and the a0 intercept of our dataset. Putting these values together will allow us to get our classification equation: 
```{r}
# Calculate a1…am
a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
a

# Calculate a0
a0 <- -model@b
a0

```
The `model@xmatrix[[1]]` accesses the support vectors in our model, while `model@coef[[1]]` accesses the coefficients of the support vectors. `model@xmatrix[[1]] * model@coef[[1]]` multiplies each support vector by its coefficient. The `colSums()` function adds up the values in each column of the resulting matrix. The resulting values are the coefficients of our classification equation and `model@b` calculates the a0 intercept of the classification equation. We see that the value of our intercept is a0 = -0.1149837


Using the coefficients and the intercept we calculated, the following is our classification equation:

<mark><u>**Classification Equation:**</u> 0.0001261907A1 + 0.0172198653A2 + 0.0320857066A3 + 0.1117659007A8 + 0.4723100583A9 - 0.2276979076A10 + 0.1724144003A11 - 0.0056522293A12 - 0.0236913365A14 + 0.0933285648A15 - 0.1149837 = 0</mark>

#### Evaluating the Performance of the Classifier
Now that we have our classification equation, we will evaluate it to see how well it classifies the data points of the dataset. We do this by using the predict function. 

```{r message=FALSE}
# See what the model predicts
pred <- predict(model, data[,1:10])
pred

# See how much of the model’s predictions match the actual classification
sum(pred == data[,11]) / nrow(data)
```
The `predict` function takes the SVM `model` that we set up earlier and applies it to `data[,1:10]`. The resulting vector `pred` shows us the predicted response variables from our model. `sum(pred == data[,11]) / nrow(data)` allows us to compare our predicted response variables to the actual response variables in our dataset. The sum of all the correct response variables predicted by our model is divided by the total number of response variables, giving us a proportion of 0.8669725. This proportion tells us how well our SVM model classifies the data points. 

<mark><u>**Classification Accuracy:**</u> 0.8669725 or 86.69725%</mark>

## K-Nearest-Neighbors (KNN) Analysis

Next, we will try a different method of classification called the k-nearest neighbors (KNN) algorithm. KNN classfies data points based on its nearest neighbors. For example, if a data point is surrounded by 5 red points and 2 blue points, it will be classified as red. In the same way, we will use this algorithm to determine whether a credit card application will be approved or not approved based on its nearest neighboring data points. 

We will use the k-nearest-neighbors classification function `kknn` contained in the R kknn package. "K" refers to the number of neighbors the algorithm will use. In order to find a value of k that will minimize discrepancies, we will try many values of k and find the one with the least amount of errors.  


```{r message=FALSE}
library(kknn)
library(readr)

data <- read_delim("C:/Users/leegu/OneDrive/Desktop/Projects/SVM/data 2.2/credit_card_data-headers.txt", delim = "\t")

# Sets a sequence of numbers from 1 to 30
k_values <- 1:30
# The vector that will store the error rates for each value of k. 
error_rates <- numeric(length(k_values))


#For each k value
for (k in k_values) {
  error_count <- 0
  #For each datapoint i
  for (i in 1:nrow(data)) {  
    model <- kknn(R1 ~ ., train = data[-i, ], test = data[i, ], k = k, scale = TRUE)
    #Predict i's class using all points except i, convert predictions to binary values (either 1 or 0)
    predicted_class <- ifelse(fitted(model) > 0.5, 1, 0) 
    actual_class <- data[i, "R1"]
    if (predicted_class != actual_class) {
      error_count <- error_count + 1
    }
  }
  #Average classification accuracy for all points i for this k
  error_rates[k] <- error_count / nrow(data)
}

least_error_index <- which.min(error_rates)
best_k <- k_values[least_error_index]

#Shows all the error rates
error_rates
#k with best accuracy
best_k


```

The outer loop `for (k in k_values)` will try k from 1 to 30. There doesn't seem to be a need to try k values greater than 30, because after a certain point the error rates keep increasing the bigger the k value gets. The inner loop `for (i in 1:nrow(data))` will iterate over each row (datapoint) in the dataset. Inside the inner loop, the knn model sets the `R1` column as the response variable, and all other columns are its predictors.  


`data[-i, ]` represents our training data, and it selects all the rows except for i. We use -i so that a data point will not use itself when searching for its nearest neighbors. `data[i, ]` is our testing data and it is essentially the opposite of 'data[-i, ]; only the row i is selected (along with its columns). 


Ultimately, our code computes two values. The `error_rates` vector shows us a matrix of how accurate each k value is by displaying the respective error rate of each k. `best_k` represents the k value with the lowest error rate, which  turned out to be for k = 12 and k = 15. Both has the same error rate of 0.1467890. This means that when we set k = 12 or k = 15, our knn model will classify the datapoints with an accuracy of 0.853211 or 85.3211%. 

<mark><u>**Ideal k-values:**</u> k = 12 and 15</mark>

<mark><u>**k-value Accuracy:**</u> 0.853211 or 85.3211%</mark>

## Conclusion

In our analysis, we created a SVM model of a data set containing credit card information. We observed a trend that our model's accuracy increases with smaller values of C. From an accuracy standpoint, our model seems to work best with a wider classifying margin.

In the context of hard classification, our wide margin would be ideal. Since no errors are preferred in hard classification, we would want a margin that is wide as possible so you can have much more confident and reliable classifications. In soft classification, where some errors and misclassifications are allowed, the large margin size of our model could be ideal or not ideal depending on the situation. In a situation where you don't want to make the mistake of a false positive classification, our wide margin would be very reliable. For example, giving out a loan to an unreliable borrower is more dire  than not giving a loan to a good borrower. Our large margin would do a great job of preventing us from giving a bad loan, despite carrying to risk of misclassifying some good borrowers as bad. In contrast, in situations where the cost of misclassification is high and precision is critical, we would want a margin that is much narrower than the one we have in this particular SVM. 

We also used the KNN algorithm to classify data points according to its nearest neighbors. We found that k = 12 and k = 15 produced the best results, with accuracy rates of approximately 85.32%.

There are a couple things we can do to further our analysis. In our `ksvm` function, we could have tested for more values of our C parameter on a greater scale. We could use loops or similar functions to thoroughly test for even bigger/smaller values and see how our model reacts. We can also test our model using different kernels. We used a linear kernel for this study, but there are certainly situations where data cannot be perfectly linearly divided. It would be intriguing to see how our model would react to various non-linear kernels. 


## Works Cited
Augmented AI. (2017, August 27). Support Vector Machine (SVM) in 7 minutes - Fun Machine Learning. YouTube. https://www.youtube.com/watch?v=Y6RRHw9uN9o&t=299s 

James, G. (n.d.). 9.2 Support Vector Classifers. In Springer Texts (2nd ed., pp. 371–377). essay, Springer. 

Schliep, K., Hechenbichler, K., & Lizee, A. (2016, March 26). Kknn: Weighted K-Nearest Neighbors. The Comprehensive R Archive Network. https://cloud.r-project.org/web/packages/kknn/kknn.pdf 
