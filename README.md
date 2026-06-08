# Predicting NFL PPR Fantasy Points with Random Forest & XGBoost
Random Forest and XGBoost machine learning models that predict season-long PPR points scored by QBs, RBs, WRs, and TEs and beat Average Draft Position ("ADP") across the board.

## Introduction
Average draft position represents the market consensus of how highly a football player is being drafted across thousands of fantasy football leagues. My goal with this project was to determine whether I could build a machine learning model that combines a player's ADP with their statistics from the previous season and their athletic measurements from the NFL Scouting Combine to outperform ADP alone as a predictor of the total number of fantasy football points (using PPR scoring) that a player will score. In short, the question is: can ADP + previous season statistics + NFL Scouting Combine data beat ADP alone as a measure of fantasy football performance?

## Brief Summary of Results
![test_mae_results](https://github.com/user-attachments/assets/a50e9fe8-9b4e-4d8e-984b-a853cd9ac70a)
![r2_results](https://github.com/user-attachments/assets/e3ce5230-b866-4bb8-bd26-dcb48d633a8c)

Both models outperformed ADP as a baseline predictor with respect to both Test MAE and Test R^2 Score. Compared to each other, the models performed similarly. However, XGBoost produced better test MAE across all positions while Random Forest produced higher R^2 scores across all positions. It's not immediately clear to me why this would be the case, so that's something I may investigate further in the future.

I was also interested in how each model's performance varied across positions. MAE is not that helpful for this type of question because, due to the nature of fantasy scoring, certain positions tend score more points than others. It's no surprise that QBs have the highest MAE and TEs the lowest, because QBs score the most points and TEs the least.

But the R^2 results are interesting. QBs decisively outperform the other positions on this metric. For most fantasy football players, this will not be surprising. QBs are generally perceived to be the least volatile position. The models seem to back this up. On the other hand, RBs are generally considered the most volatile, due in large part to the outsized effects that age, personnel changes, and injuries can have on their performance. The models back this up, as both show RBs having the lowest R^2 score, although the difference between RBs and WRs and TEs is perhaps not as pronounced as one might expect.

## Repo Tour
data_cleaning - scripts that scrape and clean data from Pro Football Reference, Fantasy Pros, and nflcombineresults.com.

exploratory_data_analysis - code to explore and refine the data.

jupyter_notebooks - Jupyter Notebooks that walk through my project step-by-step in more detail and include visualizations of the results.

models - Random Forest and XGBoost models and code for plotting and comparing model results against each other and against ADP as a baseline predictor.

## General Description of Features Used
Previous Season Statistics (e.g. passing yards, rushing yards, touchdowns, etc.)

Average Draft Position (current season and previous season)

NFL Scouting Combine Measurements (including adjusted Pro Day measurements)

## Areas for Future Improvement
Additional Features - I would like to add features like offensive line rank and draft capital to see if the models improve further.

Older Season Data - I scraped a lot of data from older seasons (pre-1987) that I ended up not using. I'd like to see how it affects the model to include those observations.

Additional Hyperparameter Tuning - I ran GridSearchCV using both my models, but only scratched the surface with tuning the hyperparameters, particularly with XGBoost. 
I'd like to experiment with other hyperparameter values using RandomizedSearchCV to see if it results in any further improvements.

## License
No license file is currently present. Treat this as all rights reserved unless a license is added. If you plan to fork/distribute, please open an issue to discuss.

