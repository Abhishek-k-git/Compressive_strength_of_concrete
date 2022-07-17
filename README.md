![Banner](https://images.unsplash.com/photo-1620902740358-c07fe4916812?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NzN8fGNvbnN0cnVjdGlvbnxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=1920&h=400&q=40)

# Compressive strength of concrete
### Concrete Compressive Strength Prediction using Machine Learning

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)

Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. The Compressive strength of concrete determines the quality of concrete. The manual testing of concretes require, making small cylinderical blocks of concretes with different proportions of materials and testing against compression. This exhausts both time and money.

One way of reducing the wait time and reducing the number of combinations to try is to make use of digital simulations, where we can provide information to the computer about various dependent parameters and the computer tries different combinations to predict the compressive strength. This way we can reduce the number of combinations we can try physically and reduce the amount of time for experimentation. But, to design such software we have to know mathematical relations between different parameters and run simulations based on these equations, but we cannot expect the relations to be same in real-world. *Since, these tests have been performed for many numbers of times now and we have enough real-world data that can be used for **predictive modelling**.*


> **Implemented Algorithms:**
1. Random Forest Regressor
2. Gradient Boosting Regressor
3. AdaBoosting Regressor
4. KNeighbors Regressor
5. Bagging Regressor
6. Support vector regressor
7. XG Boost Regressor
8. Decision Tree Regressor


> **Problem Statement:**
To predict compressive strength of concrete against various incredients (for different proportions) like cement, water, superplasticizer, coarseagg, fineagg etc. Then predict the accuracy of following algorithms.

We know that data is messy. A dataset may contain multiple missing values. In that situation, we have to clean the dataset. To avoid this kind of hassle we are going to use a pre-cleaned dataset. You can download the dataset (.CSV file) from [here]([https://archive.ics.uci.edu/ml/datasets/banknote+authentication](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength))

``` pd.read_csv('data.csv') ``` - convert to pandas dataframe

``` data.isna().sum() ``` - check wheather this dataset contains any empty/null value or

> **Data visualization:**

![dataset](https://github.com/Abhishek-k-git/Compressive_strength_of_concrete/blob/main/images/data_stats.png)

After dataprocessing or cleaning, it is very crucial to visualize dataset, there are many datavisualization tool out there. But here we use [seaborn](https://seaborn.pydata.org/), which is a python data visualization library based on matplotlib and [graphviz](https://graphviz.org/), which is an open-source python module that is used to create graph objects which can be completed using different nodes and edges. It provides a high-level interface for drawing attractive and informative statistical graphics.

> **Training / Testing data split:**

Now data is divided into two sets one is *training dataset* which is used to train the model (just like a new born child learns by sensing things around him), the other dataset is *testing dataset* which is used to evaluate or predict the accuracy of model. The machine uses its model, apply to testing dataset to give out predicted results. The predicted output then compared to final result in actual dataset (In this case it is labeled as *class*). That's why it is necessary to first drop that column named class, before we train our model.

```
X = data.drop('class', axis = 1)
y = data['class']

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)
```

> **Data Modelling**
#### 1. Random Forest Regressor
A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
```
model = RandomForestRegressor()
model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 1 |	Random Forest Regressor	| 0.895577 |

**KFold cross validation**
```
k = 20
kfold = KFold(n_splits = k, random_state = 10, shuffle=True)
k_result = cross_val_score(model, X, y, cv = kfold)
k_accuracy = np.mean(abs(k_result))
#k_accuracy

data = pd.DataFrame({'Algorithm': ['Random Forest Regressor KFold'], 
                    'Accuracy': k_accuracy}, index = {1})

results = pd.concat([results, data])
results = results[['Algorithm', 'Accuracy']]
results
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 2 |	Random Forest Regressor KFold |	0.916420 |


#### 2. Gradient Boosting Regressor
```
model = GradientBoostingRegressor()
model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 3 |	GradientBoosting Regressor | 0.895237 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 4 |	GradientBoosting Regressor KFold | 0.902362 |


#### 3. AdaBoosting Regressor
```
model = AdaBoostRegressor()
model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 5 |	AdaBoosting Regressor | 0.777865 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 6 |	AdaBoosting Regressor KFold |	0.783261 |


#### 4. KNeighbors Regressor
```
model = KNeighborsRegressor()
model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 7 |	KNeighbors Regressor | 0.67677 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 8 |	KNeighbors Regressor KFold | 0.711765 |


#### 5. Bagging Regressor
```
model = BaggingRegressor()
model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 9 |	Bagging Regressor |	0.877553 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 10 | Bagging Regressor KFold | 0.904520 |


#### 6. Support vector regressor
```
model = SVR(kernel = 'linear')
model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 11 | Support vector regressor |	0.481796 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 12 | Support vector regressor KFold	| 0.550172 |


#### 7. XG Boost Regressor
```
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
```
```
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 13 | XG Boost Regressor	| 0.919621 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 14 | XG Boost Regressor KFold |	0.929891 |


#### 8. Decision Tree Regressor
```
dec_model = DecisionTreeRegressor()
dec_model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 15 | DecisionTree Regressor |	0.822001 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 16 | DecisionTree Regressor KFold	| 0.864506 |

**Feature elimination**
feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached.

#### 9. New Decision Tree Regressor
```
new_dec_model = DecisionTreeRegressor()
new_dec_model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 17 | new DecisionTree Regressor	| 0.693405 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 18 | new DecisionTree Regressor KFold	| 0.858617 |

![map](https://github.com/Abhishek-k-git/Compressive_strength_of_concrete/blob/main/images/map.png)

#### 10. pruned decision tree
```
new_dec_model = DecisionTreeRegressor()
new_dec_model.fit(x_train, y_train)
```
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 19 | pruned decision tree	| 0.660811 |

**KFold cross validation**
| S.No | Algorithm | Accuracy |
| :--: | :------: | :----------: |
| 20 | pruned decision tree KFold	| 0.663904 |

>> ```XG Boost Regressor KFold``` is best performing algorithm out of these with an accuracy of ```0.929891```
