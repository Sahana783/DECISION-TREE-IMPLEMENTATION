# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SAHANA

*INTERN ID*: CT04DZ1200

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

##DESCRIPTION OF THE TASK 1: DECISION TREE IMPLEMENTATION

The provided Python code represents a complete and practical implementation of a Decision Tree Classification model using the widely known Iris dataset. This notebook-style code is designed for educational 
and internship purposes, aiming to introduce the learner to fundamental concepts in machine learning, particularly classification problems, data visualization, model training, and evaluation. It makes use 
of several essential Python libraries, most notably Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn. Each library contributes specific functionality essential to the machine learning workflow.

The code begins by importing core libraries. NumPy is used for numerical operations on arrays, which underlie most data structures in machine learning. Pandas enables the creation and manipulation of structured data using DataFrames. This format is particularly useful for understanding and visualizing data in tabular form. Matplotlib.pyplot and Seaborn are visualization libraries that help in creating plots, graphs, 
and charts, which are crucial for understanding data distributions and model decisions. The sklearn.datasets module provides the Iris dataset, which is used as a sample for classification. train_test_split from sklearn.model_selection is used to divide the dataset into training and testing subsets, a standard practice in machine learning to evaluate how well the model generalizes to new data. The DecisionTreeClassifier from sklearn.tree is the core algorithm used to create the decision tree model, and plot_tree allows visualizing this model. Metrics such as accuracy_score, classification_report, and confusion_matrix from sklearn.metrics are used to evaluate the performance of the trained model.

The Iris dataset itself is a classical dataset in machine learning and statistics. It consists of 150 samples of iris flowers with four features per sample: sepal length, sepal width, petal length, and petal width. The dataset has three target classes corresponding to three species of iris flowers: Setosa, Versicolor, and Virginica. It is well-suited for beginners as it is small, well-balanced, and easy to visualize.

After loading the dataset using load_iris(), the code extracts the feature matrix X and the target vector y. These represent the input variables and the class labels, respectively. To make the data easier to understand and manipulate, it is converted into a Pandas DataFrame. Each row in the DataFrame represents one sample, and the columns represent the respective features. Two additional columns are added: one for the numeric target class and one for the species name mapped from the target. Displaying the first few rows using df.head() gives the user a quick preview of the data structure, which is helpful for validation and understanding.

The next step involves examining the class distribution using value_counts(). This ensures that the dataset is balanced and that there are no biases in the data. This is important in classification problems because a highly imbalanced dataset can lead the model to favor one class over another, resulting in misleading accuracy scores.

Following this, the dataset is split into training and testing sets using the train_test_split() function. This is a standard procedure in machine learning that helps assess the model’s ability to generalize to new, unseen data. In this case, 70% of the data is used for training and 30% for testing. The random_state=42 argument ensures reproducibility, meaning that every time the code is run, the data is split in the same way.

Once the data is split, a decision tree classifier is created using the DecisionTreeClassifier() class with the gini criterion. The Gini index is a measure of node impurity used in classification trees.
By setting max_depth=3, the tree is restricted to a depth of three levels, preventing it from becoming overly complex or overfitting to the training data. The model is trained using the fit() method, 
which takes the training data and learns patterns in it.

Predictions are then made on the test set using the predict() method. The predicted classes are compared to the actual classes in the test set to evaluate the model. The accuracy_score measures the proportion of correct predictions. The classification_report provides a more detailed analysis, including precision (how many selected items are relevant), recall (how many relevant items are selected), F1-score (a harmonic mean of precision and recall), and support (the number of actual occurrences of each class). The confusion_matrix is another important metric that gives a matrix of actual vs. predicted labels, showing exactly how the model is performing across different classes.

One of the most valuable aspects of this code is the visualization of the decision tree itself using the plot_tree() function. This function plots the structure of the decision tree, showing how it splits the dataset based on feature values at each node. It displays the Gini index, the number of samples in each node, and the class distribution. The tree is filled with colors representing different classes, making 
it easier to interpret. This visualization helps understand the logic behind the decision-making process of the classifier, making it a powerful tool for model explainability and debugging.

In addition to the model visualization, the code includes a Seaborn pairplot() which visualizes the distribution of feature values across different classes. This plot creates scatter plots between all pairs of features and colors the points by species, helping to visually identify which features are most useful for classification. This step, while optional, is extremely helpful in exploratory data analysis (EDA) and can guide feature selection or engineering in more complex datasets.

This code has broad applications beyond just the Iris dataset. The techniques and structure used here are foundational to building machine learning models in various fields. For example, in healthcare, 
decision trees can be used to predict whether a patient has a certain disease based on clinical symptoms. In finance, they can help determine if a transaction is fraudulent based on transaction characte
ristics. In education, decision trees might classify student performance based on study habits and attendance. Because decision trees are interpretable, they are often used in regulated industries where 
model decisions need to be explainable.

In conclusion, the aim of the code is to provide a complete, understandable, and reproducible example of using a decision tree classifier for a multi-class classification task. It utilizes popular Python libraries, follows industry-standard machine learning workflow steps, and emphasizes model interpretability and data visualization. While the dataset is simple, the methodology applies to much more complex
real-world problems, and the techniques used here are foundational for further study and application in data science and artificial intelligence. The code not only teaches how to build a machine learning 
model but also how to understand and trust it, which is a critical part of responsible AI development.


##OUTPUT
<img width="1336" height="642" alt="Image" src="https://github.com/user-attachments/assets/c8e87a54-8f19-45fd-aafa-a78b517b275c" />

<img width="1057" height="627" alt="Image" src="https://github.com/user-attachments/assets/872c26bc-b912-4b96-93cd-eb6ff844e01d" />

<img width="1676" height="687" alt="Image" src="https://github.com/user-attachments/assets/f1118f5c-eedc-4d47-9142-2e9c61058b7c" />

<img width="1275" height="626" alt="Image" src="https://github.com/user-attachments/assets/41f43254-2e41-49db-b8a9-e1baae7b6462" />

<img width="1611" height="552" alt="Image" src="https://github.com/user-attachments/assets/6ad53872-7c42-41bf-8665-fc6d1d20ebd9" />

<img width="1752" height="656" alt="Image" src="https://github.com/user-attachments/assets/8e1d0f7b-ff07-4953-9c6a-9ed593d91240" />

<img width="1640" height="577" alt="Image" src="https://github.com/user-attachments/assets/98883a06-4e53-4ca1-859c-1ad5180da3a1" />

<img width="1675" height="346" alt="Image" src="https://github.com/user-attachments/assets/bae94179-142e-469d-bf48-dd074f759ad2" />

<img width="1688" height="365" alt="Image" src="https://github.com/user-attachments/assets/ef45e014-57cc-40b8-926e-9154403f23ed" />
