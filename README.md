# MLOps Salary Prediction Project

This project demonstrates a simple end-to-end Machine Learning workflow as part of an MLOps assignment.  
The goal is to predict salary levels using cleaned job-related data and evaluate model performance.

## Project Structure

├── main.py  
├── salary_data_cleaned.csv  
├── README.md  
├── .gitignore  

## Dataset
The project uses a cleaned salary dataset derived from job listings.

Target variable: avg_salary
Features include:
- Job title
- Company details
- Location
- Industry information
- Other relevant attributes
  
## Workflow

1. Load cleaned dataset  
2. Perform train-test split  
3. Convert salary into binary classes (low / high)  
4. Train Logistic Regression model  
5. Evaluate using:
   - Confusion Matrix
   - Accuracy
   - Precision, Recall, F1-score  
## Model Used
- Logistic Regression (Classification)
Salary classes are created using the median salary:
- 0 → Below median salary
- 1 → Above or equal to median salary
  
## Performance Metrics
The model is evaluated using:
- Confusion Matrix
- Accuracy Score
- Classification Report
These metrics help understand prediction correctness and error distribution.

## How to Run

Activate virtual environment (if not already active):

