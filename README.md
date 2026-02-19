Project Title: Spam vs Ham Email Classifier using NLP & Machine Learning

The dataset used in this project is an SMS spam collection containing two columns: a label indicating whether the message is ham or spam, and the message text itself. Initial data exploration helps determine the dataset size and understand the distribution of ham versus spam messages. This step is part of Exploratory Data Analysis (EDA), which is essential for identifying class imbalance, spotting irregularities, and planning appropriate preprocessing and modeling strategies.

During preprocessing, the message text is treated as the input feature (X), while the labels are converted into numerical form (y), where ham is mapped to 0 and spam is mapped to 1. 
	Preprocessing ensures that the data is structured, standardized, and ready for numerical transformation, which is a core    requirement in NLP pipelines.
The dataset is then divided into training and testing subsets using an 80–20 split. This separation is crucial to prevent overfitting and to ensure that the model generalizes well to real-world data, making train–test splitting a foundational concept in machine learning validation.
Since machine learning models require numerical input, the textual messages are transformed into vectors using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization. TF-IDF assigns importance to words based on how frequently they appear in a specific message and how rare they are across the entire dataset. 

For classification, the Multinomial Naive Bayes algorithm is used because it performs exceptionally well on text data and word-frequency features. The model is based on Bayes’ Theorem and predicts the probability of each class given the words in a message, assuming independence between words. 
Despite this simplifying assumption, Naive Bayes is widely used in real-world spam detection due to its speed, simplicity, and strong performance on moderate-sized datasets.

Model performance is evaluated using several important metrics. 
-->	Accuracy measures the overall proportion of correct predictions.
-->	Precision indicates how many messages predicted as spam are truly spam, which is important when minimizing false spam alerts.
-->	Recall measures how many actual spam messages are successfully detected, which is critical for security and filtering systems.
--> F1-score balances precision and recall, making it one of the most reliable overall indicators of classification quality. Together, these metrics provide a comprehensive understanding of model effectiveness.

A confusion matrix further explains prediction behavior by displaying true positives, true negatives, false positives, and false negatives. 
Overall, this project builds foundational skills in NLP, machine learning, and data science. It demonstrates text preprocessing, TF-IDF feature engineering, probabilistic classification with Naive Bayes, proper evaluation using multiple metrics, and visualization for interpretability. These concepts directly translate to real-world applications such as email spam filtering, SMS fraud detection, customer feedback classification, toxic comment moderation, and automated content categorization.
 
