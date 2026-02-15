---
layout: post
author: Wang Ming
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background

### Business Background
Mobile applications are critical customer touchpoints for sportswear brands such as Nike, Adidas, PUMA, and Gymshark. Poor app experiences can result in low ratings, negative reviews, reduced engagement, and lower user retention.
Although user reviews on the Google Play Store provide rich, unstructured feedback, manual analysis is time-consuming and often fails to fully leverage actionable insights. Therefore, a systematic, data-driven approach is needed to better understand user concerns and satisfaction drivers.

### Business Goal
Enhance user satisfaction and engagement for sportswear mobile applications by extracting actionable insights from Google Play Store reviews.

### Project Objectives
1.	Topic Modelling – Identify key themes and recurring discussion topics across brands.
2.	Sentiment Analysis – Evaluate user sentiment and emotional tone in app reviews.
3.	Predictive Modelling (My Focus) – Determine the primary drivers of user satisfaction through statistical analysis and predictive modelling.

### My Objective
My focus is to identify the key drivers of user satisfaction using predictive modelling techniques. By analyzing review data and related factors, I aim to determine what most strongly influences positive and negative user experiences.
The ultimate goal is to provide prioritized, data-driven recommendations that support targeted app improvements and enhance overall user satisfaction.

## Work Accomplished

### Data Preparation

#### 1. Initial Dataset Characteristics

The dataset initially comprised **6446 rows** and **8 columns**. The columns were a mix of `object` types (`brand`, `review_id`, `at`, `content`, `reply_content`, `review_created_version`) and `int64` types (`score`, `thumbs_up`). The `at` column was identified as needing conversion to `datetime` type.

#### 2. Key Observations from Data Inspection

*   **`reply_content`**: This column had a very high percentage of missing values (93.25%), rendering it unsuitable for direct analysis. This indicated that most reviews did not receive a reply.
*   **`review_created_version`**: Exhibited a notable percentage of missing values (10.55%). Its relevance for predicting user satisfaction based on review content and metadata was considered limited.
*   **`review_id`**: A unique identifier for each review, but held no direct predictive power.
*   **`thumbs_up`**: This numerical column was highly skewed, with 82% of reviews having 0 thumbs-up, suggesting limited direct utility for the classification task without significant transformation.

#### 3. Data Cleaning Steps

Based on the initial inspection and the objective of predicting user satisfaction, the following columns were dropped to streamline the dataset and remove noise:

*   **`review_id`**: Dropped as it's a unique identifier with no predictive value.
*   **`reply_content`**: Removed due to the overwhelming majority of missing values (>93%) and the difficulty in imputing or meaningfully using such sparse data.
*   **`thumbs_up`**: Dropped to simplify the feature set, focusing on more direct drivers of satisfaction. Its highly skewed distribution also presented challenges.
*   **`review_created_version`**: Dropped due to a significant number of missing values (10.55%) and its less direct relevance to the sentiment or satisfaction category.
*   The `at` column was converted to `datetime` type for time-based feature extraction.

#### 4. Feature Engineering Steps

Several new features were engineered to provide more granular insights and improve the classification models:

*   **`review_length` (Word Count)**:
    *   **Derivation**: Calculated by counting the number of words in the `content` column.
    *   **Rationale**: Longer reviews often indicate stronger opinions (either highly satisfied or dissatisfied) and more detailed feedback, particularly for negative experiences.

*   **`rating_category` (Categorized Score)**:
    *   **Derivation**: Transformed the numerical `score` (1-5 stars) into three ordinal categories:
        *   'Low': Scores of 1 or 2
        *   'Medium': Score of 3
        *   'High': Scores of 4 or 5
    *   **Rationale**: This simplifies the target variable for a classification approach, aligning with the bimodal distribution of scores and making business decisions more actionable (e.g., prioritizing 'Low' reviews).

*   **`time_category` (Part of Day)**:
    *   **Derivation**: Extracted from the `at` (timestamp) column, categorizing the hour into 'Night' (0-5 AM), 'Morning' (6-11 AM), 'Afternoon' (12-5 PM), and 'Evening' (6-11 PM).
    *   **Rationale**: Helps identify if user sentiment varies based on the time of day, potentially revealing patterns related to app usage context or mood.

*   **`day_of_month`**:
    *   **Derivation**: Extracted the day number (1-31) from the `at` column.
    *   **Rationale**: To identify if specific days of the month correlate with review patterns, possibly linked to billing cycles or recurring events.

*   **`month`**:
    *   **Derivation**: Extracted the month number (1-12) from the `at` column.
    *   **Rationale**: To capture seasonal trends or correlations with monthly events like holidays or product launches.

*   **`week_of_year`**:
    *   **Derivation**: Extracted the week number (1-52/53) from the `at` column.
    *   **Rationale**: Provides a finer temporal resolution than 'month' to detect weekly patterns or events influencing feedback.

*   **`day_of_week`**:
    *   **Derivation**: Extracted the day of the week (e.g., Monday, Tuesday) from the `at` column.
    *   **Rationale**: To explore if user behavior and satisfaction differ across weekdays and weekends or specific days of the week.

### Modelling

#### Classification Approach

The problem of identifying key drivers of user satisfaction from app reviews was framed as a **classification task** rather than a regression problem. This decision was driven by several key factors:

1.  **Direct Actionability**: Predicting categories like 'Low', 'Medium', or 'High' satisfaction is more directly actionable for product teams. For instance, 'Low' category reviews warrant urgent attention, while 'High' category reviews indicate successful features.
2.  **Handles Bimodal Distribution**: The initial exploratory data analysis (EDA) revealed a distinct bimodal distribution of review scores (peaks at 1 and 5 stars). Classification models are well-suited to learn and predict these distinct categories of satisfaction.
3.  **Clearer Interpretation**: Categorical predictions offer unambiguous results, simplifying communication and decision-making compared to an imprecise fractional score.
4.  **Optimized for Prioritization**: This approach allows for specific optimization to accurately identify critical 'Low' rated reviews, ensuring important feedback is not overlooked.

The target variable (`y`) for this classification task is `rating_category`, which was engineered from the original `score` into three classes: 'Low' (scores 1-2), 'Medium' (score 3), and 'High' (scores 4-5). Features used for modeling include `brand`, `review_length`, `hour`, `day_of_month`, `month`, `week_of_year`, and `day_of_week`, which were preprocessed using one-hot encoding for categorical features.

#### Implemented Classification Models

To address the classification task, three different machine learning algorithms were implemented and evaluated:

1.  **RandomForestClassifier**: An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is robust against overfitting and capable of handling non-linear relationships.
2.  **DecisionTreeClassifier**: A fundamental tree-based model that creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. While simpler, it provides a baseline for more complex tree-based methods.
3.  **LGBMClassifier (Light Gradient Boosting Machine)**: A gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient, making it suitable for large datasets and known for its speed and accuracy.

#### Hyperparameter Tuning Process

For each implemented model, **hyperparameter tuning** was performed using **GridSearchCV** with **5-fold cross-validation**. This systematic search approach explores a specified range of hyperparameter values to find the combination that yields the best model performance. The key aspects of the tuning process were:

*   **Cross-Validation**: Using `cv=5`, the dataset was split into 5 folds, with the model trained on four folds and evaluated on the remaining one, rotating through all folds. This helps in obtaining a more robust estimate of model performance and prevents overfitting to a single train-test split.
*   **Scoring Metric**: The optimization metric for tuning was `'f1_weighted'`. The F1-score is the harmonic mean of precision and recall, providing a balanced measure that is particularly useful for imbalanced datasets, which can be common in satisfaction ratings where 'High' ratings might outnumber 'Low' or 'Medium' ones. The 'weighted' average accounts for class imbalance by computing metrics for each label and finding their average weighted by support (the number of true instances for each label).
*   **Parameter Grids**: Specific parameter grids were defined for each model, specifying the hyperparameters and their candidate values to be searched (e.g., `n_estimators`, `max_depth`, `learning_rate`). The `n_jobs=-1` parameter was used to utilize all available CPU cores for parallel processing, speeding up the tuning process.

### Evaluation

Present the performance metrics for each model on the test set, compare their effectiveness, and highlight the best-performing model and its top feature importance.

#### Model Performance Overview

The performance of the hyperparameter-tuned classification models was assessed on the test set. The following metrics were used to evaluate their effectiveness: Accuracy, Precision (weighted), Recall (weighted), and F1-Score (weighted).

| Model                  | Accuracy | Precision (weighted) | Recall (weighted) | F1-Score (weighted) |
| :--------------------- | :------- | :------------------- | :---------------- | :------------------ |
| RandomForestClassifier | 0.8000   | 0.7534               | 0.8000            | 0.7721              |
| DecisionTreeClassifier | 0.7620   | 0.7241               | 0.7620            | 0.7378              |
| LGBMClassifier         | 0.7984   | 0.7535               | 0.7984            | 0.7718              |


#### Best Performing Model and Feature Importances

Based on the F1-weighted score, the **RandomForestClassifier** was identified as the best-performing model, achieving an F1-weighted score of **0.7721**.

The top 5 most important features for the RandomForestClassifier in predicting user satisfaction categories are:

1.  **review_length**: 0.4103
2.  **hour**: 0.1345
3.  **week_of_year**: 0.1112
4.  **day_of_month**: 0.1067
5.  **month**: 0.0635

## Recommendation and Analysis

The top 5 feature importances for the RandomForestClassifier model provides crucial insights into the factors most influencing user satisfaction (i.e., `rating_category`).

1.  **`review_length` (Importance: ~0.41)**:
    *   **Insight**: This is by far the most significant predictor of user satisfaction. As previously noted in the EDA, longer reviews tend to correlate with lower scores. This reinforces the idea that users who are highly dissatisfied or encounter significant issues are more likely to elaborate on their experiences.
    *   **Actionable Recommendation**: Implement a system to flag and prioritize longer reviews, especially those identified as 'Low' or 'Medium' satisfaction categories by the model, for immediate human review. These reviews likely contain detailed feedback on pain points that require urgent attention or deeper investigation.

2.  **`hour` (Importance: ~0.13)**:
    *   **Insight**: The specific hour of the day a review is posted is the second most important feature. This suggests that user sentiment might be influenced by the time they are using the app or writing the review. For instance, reviews posted during specific hours (e.g., late night, early morning) might lean towards particular sentiments, possibly due to usage patterns or frustration experienced during off-peak support hours.
    *   **Actionable Recommendation**: Investigate the distribution of 'Low' ratings across different hours of the day. If certain hours show a disproportionately high number of negative reviews, this could point to issues related to app performance during peak usage, lack of immediate support, or specific user scenarios occurring at those times. Adjust support availability or app features accordingly.

3.  **`week_of_year` (Importance: ~0.11)**:
    *   **Insight**: The week of the year also plays a notable role. This indicates potential seasonal trends or correlations with specific events (e.g., holidays, marketing campaigns, major app updates) that occur during certain weeks and impact user satisfaction.
    *   **Actionable Recommendation**: Correlate weeks with significantly lower (or higher) average satisfaction scores with historical app updates, marketing activities, or external events. This can help identify successful campaigns or problematic releases, allowing for better timing of future initiatives.

4.  **`day_of_month` (Importance: ~0.10)**:
    *   **Insight**: Similar to `week_of_year`, the day of the month also influences satisfaction. This could be related to monthly billing cycles, specific feature releases that happen at the beginning or end of a month, or general user behavior patterns.
    *   **Actionable Recommendation**: Analyze review scores around critical dates in the month, such as billing dates or planned maintenance schedules. If specific days consistently show drops in satisfaction, investigate app performance or communication strategies around those times.

5.  **`month` (Importance: ~0.06)**:
    *   **Insight**: The overall month of the year has a smaller but still significant impact on satisfaction. This provides a broader view of seasonal trends than `week_of_year`.
    *   **Actionable Recommendation**: Identify months with consistently lower satisfaction and cross-reference with major seasonal events, holidays, or business-specific cycles. This can help in proactively addressing potential issues or preparing for periods of increased user dissatisfaction.

### Recommendations:

*   **Focus on Content Analysis**: Given the high importance of `review_length`, deep diving into the content of longer, low-rated reviews using NLP techniques is critical to pinpoint specific problems. This model serves as an excellent filter for identifying reviews that warrant such in-depth analysis.
*   **Time-Based Monitoring**: Implement dashboards and alerts to monitor user satisfaction trends based on the `hour`, `day_of_month`, `week_of_year`, and `month`. Anomalies in these time-based metrics should trigger investigations into corresponding operational or app-related events.
*   **Iterative Improvement**: Use these insights to inform an iterative improvement cycle. Address the issues identified from longer reviews and time-based patterns, then monitor changes in satisfaction metrics and feature importances over time to validate the impact of interventions.

## AI Ethics

Discuss potential data science ethics issues relevant to the project, covering privacy, fairness, accuracy, accountability, and transparency.

### Privacy
This project involves analyzing user-generated review content, which, even if anonymized, can sometimes contain indirectly identifiable information or sensitive personal opinions. Ethical considerations include:
*   **Data Minimization**: Ensuring only necessary data is collected and processed.
*   **Anonymization/Pseudonymization**: Implementing robust techniques to protect user identities. Although `review_id` was dropped, the content itself could contain personally identifiable information.
*   **Consent**: While app reviews are typically public, using them for predictive modeling (especially for internal business decisions) should align with the terms of service and user expectations regarding data usage.
*   **Data Security**: Protecting the dataset from breaches, as it contains user feedback that could be exploited.

### Fairness
Biases can inadvertently enter the model at various stages:
*   **Sampling Bias**: If the review data itself is not representative of all user groups (e.g., certain demographics are less likely to leave reviews, or review content varies systematically across groups), the model might reflect and amplify these biases.
*   **Model Bias**: The model might disproportionately flag reviews from certain demographics or user groups as 'Low' satisfaction due to underlying patterns in their language or usage behavior, leading to unfair targeting for interventions or ignoring their valid feedback. For instance, if certain language styles are more common in reviews from particular regions, the model might misinterpret them. The current model does not include demographic features, but even proxy features like `time_category` or `brand` could correlate with demographic groups.
*   **Mitigation**: Regularly audit the model's performance across different user segments if such data were available. Ensure that recommendations derived from the model do not lead to discriminatory outcomes.

### Accuracy
Model accuracy is crucial for effective and ethical decision-making:
*   **Importance of Correct Categorization**: Misclassifying a 'Low' satisfaction review as 'High' could mean critical bugs or user pain points are overlooked, leading to user dissatisfaction and churn. Conversely, misclassifying a 'High' review as 'Low' could lead to misdirected efforts and wasted resources.
*   **Nuance of Human Sentiment**: User reviews are complex; a classification model, especially one based on limited metadata features, might struggle to capture sarcasm, irony, or highly nuanced sentiment. The bimodal distribution of scores suggests users often express strong opinions, but even within these extremes, there can be subtle variations that a simplified 'Low', 'Medium', 'High' categorization might miss.
*   **Continuous Improvement**: The model needs continuous monitoring and retraining with new data to maintain its accuracy and adapt to evolving user language and app functionalities.

### Accountability
Accountability ensures responsibility for the model's impact:
*   **Ownership of Decisions**: Who is accountable when the model makes a mistake, e.g., a critical bug mentioned in a 'Low' review is missed because the model misclassified it? The responsibility typically lies with the product owner or the data science team deploying the model.
*   **Human Oversight**: The model's recommendations, especially for 'Low' satisfaction reviews, should ideally be reviewed by human analysts. Automated responses or actions based solely on model predictions without human verification could lead to inappropriate or alienating interactions with users.
*   **Documentation**: Thorough documentation of the model's design, training process, limitations, and performance metrics is essential for accountability and auditability.

### Transparency
Transparency helps build trust and enables effective use of the model's insights:
*   **Model Interpretability**: Tree-based models like RandomForestClassifier and DecisionTreeClassifier offer a degree of interpretability through feature importances. This is vital for understanding *why* a review was classified in a certain way. For example, knowing that `review_length` is the most important feature allows product teams to focus on longer reviews for detailed feedback.
*   **Explainability of Recommendations**: When the model suggests prioritizing certain reviews or flagging specific time-based patterns, the underlying reasons (e.g., higher likelihood of 'Low' satisfaction for longer reviews posted at night) should be clearly communicated. This helps stakeholders trust and act upon the recommendations.
*   **Communicating Limitations**: Being transparent about what the model can and cannot do, and the potential biases or inaccuracies, is crucial. This manages expectations and prevents over-reliance on automated predictions.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. https://github.com/brotherbear9004/ITD214_project_data
Dataset and Python code.
[8090787y_itd214_project.zip](https://github.com/user-attachments/files/25322316/8090787y_itd214_project.zip)

