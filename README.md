# Predicting NFL PPR Fantasy Points with Random Forest & XGBoost
Random Forest and XGBoost machine learning models for predicting season-long PPR points scored by QBs, RBs, WRs, and TEs.

# Repo Tour
data_cleaning - scripts that scrape and clean data from Pro Football Reference, Fantasy Pros, and nflcombineresults.com.

exploratory_data_analysis - code to explore and refine the data.

jupyter_notebooks - Jupyter Notebooks that walk through my project step-by-step in more detail and include visualizations of the results.

models - Random Forest and XGBoost models and code for plotting and comparing model results against each other and against ADP as a baseline predictor.

# General Description of Features Used
Previous Season Statistics (e.g. passing yards, rushing yards, touchdowns, etc.)

Average Draft Position (current season and previous season)

NFL Scouting Combine Measurements (including adjusted Pro Day measurements)

# Brief Summary of Results
Both models outperformed ADP as a baseline predictor with respect to both Test MAE and Test R^2 Score:
![test_mae_results](https://github.com/user-attachments/assets/a50e9fe8-9b4e-4d8e-984b-a853cd9ac70a)
![r2_results](https://github.com/user-attachments/assets/e3ce5230-b866-4bb8-bd26-dcb48d633a8c)

# Areas for Future Improvement
Additional Features - I would like to add features like offensive line rank and draft capital to see if the models improve further.

Older Season Data - I scraped a lot of data from older seasons (pre-1987) that I ended up not using. I'd like to see how it affects the model to include those observations.

Additional Hyperparameter Tuning - I ran GridSearchCV using both my models, but only scratched the surface with tuning the hyperparameters, particularly with XGBoost. 
I'd like to experiment with other hyperparameter values using RandomizedSearchCV to see if it results in any further improvements.

## License
No license file is currently present. Treat this as all rights reserved unless a license is added. If you plan to fork/distribute, please open an issue to discuss.

