import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer  
from xgboost import XGBRegressor 


def create_mask(feature, df, stat):
    """Identify rows that have a 0 value for the specified feature but nonzero value
       for passing, rushing, or receiving yards (as specified).

    Args:
        feature: String representing the feature to search for 0 values.
        df:      A Pandas dataframe containing NFL statistics.
        stat:    String that is either 'Pass', 'Rushing', or 'Receiving'. Specifies the 
                 yardage feature to examine.

    Returns:
        Boolean mask that identifies the rows of df where the feature is 0 but the specified
        yardage type is nonzero.
    """
    return ((df[feature] == 0) & (df[f'{stat}_Yds_prev'] > 0))


def fix_0_values(df):
    """Replace erroneous 0 values with NaNs in the provided dataframe.
    
    Args:
        df: A Pandas dataframe containing NFL statistics.

    Returns:
        Dataframe with 0 values replaced with NaNs in the problematic features.
    """
    # These are the advanced stats that need to be fixed. Fumbles Lost and Age are
    # addressed separately below.
    features_to_fix = {
        'Pass': ['Pass_Succ%_prev', 'Pass_1D_prev', 'QBR_prev'],
        'Rushing': ['Rushing_Succ%_prev', 'Rushing_1D_prev'],
        'Receiving': [
            'Receiving_1D_prev',  'Receiving_Ctch%_prev', 'Receiving_Succ%_prev',
            'Receiving_Tgt_prev', 'Receiving_Y/Tgt_prev'
        ]
    }

    for stat_type in features_to_fix:
        for ftr in features_to_fix[stat_type]:
            mask = create_mask(ftr, df, stat_type)
            # Locate the rows where the mask is true and replace the specified feature
            # values with 0's.
            df.loc[mask, ftr] = np.nan

    fl_mask = (
        (df['Fumbles_FL_prev'] == 0) &
        (df['Fumbles_prev'] > 0)      
    )
    df.loc[fl_mask, 'Fumbles_FL_prev'] = np.nan

    age_mask = ((df['Age_prev'] == 0))
    df.loc[age_mask, 'Age_prev'] = np.nan

    return df


def get_train_test_split(pos):
    """Create train/test split for the specified position.

    Args:
        pos: A string that is either 'QB', 'RB', 'WR', or 'TE' (case insensitive).

    Returns:
        Training and test datasets for the specified position.
    """
    df = pd.read_csv(f'{pos}_with_prev.csv')
    df = fix_0_values(df)

    features = []
    # These are the features that I want my model to include regardless of the position.
    base_features = [
        'Age',              'ADP',              'Height(in)',         'Weight(lbs)',      
        'Hand Size(in)',    'Arm Length(in)',   'Wonderlic',          '40Yard',           
        'Bench Press',      'Vert Leap(in)',    'Broad Jump(in)',     'Shuttle',
        '3Cone',            'ADP_prev',         'Fumbles_FL_prev',    'G_prev',           
        'GS_prev',          'Fantasy_PPR_prev'
    ]
    # Additional features to include for quarterbacks.
    pass_features = [
        '4QC_prev',         'Cmp_prev',         'Cmp%_prev',          'GWD_prev',         
        'Int_prev',         'Pass_1D_prev',     'Pass_ANY/A_prev',    'Pass_AY/A_prev',     
        'Pass_Att_prev',    'Pass_Lng_prev',    'Pass_NY/A_prev',     'Pass_Succ%',       
        'Pass_TD_prev',     'Pass_TD%_prev',    'Pass_Y/A_prev',      'Pass_Y/C_prev',    
        'Pass_Y/G_prev',    'Pass_Yds_prev',    'Passing_Rk_prev',    'QBR_prev',
        'Rate_prev',        'Int%_prev',        'Sk_prev',            'Sk%_prev'
    ]
    # Additional rushing features. I am likely going to include these for all positions, but
    # listing them separately in case I want to remove them when analyzing WRs and TEs (because 
    # they are less relevant for those positions).
    rush_features = [
        
        'Rushing_Rk_prev',  'Rushing_Succ%_prev', 'Rushing_TD_prev',  'Rushing_Y/A_prev', 
        'Rushing_Y/G_prev', 'Rushing_Yds_prev'
    ]
    # Additional receiving features to include for every position except quarterbacks.
    receiving_features = [
        'Receiving_Y/Tgt_prev', 'Receiving_Yds_prev', 'Receiving_Y/R_prev',   'Receiving_Y/G_prev',
        'Receiving_Tgt_prev',   'Receiving_TD_prev',  'Receiving_Succ%_prev', 'Receiving_Rk_prev',
        'Receiving_Rec_prev',   'Receiving_R/G_prev', 'Receiving_Lng_prev',
        'Receiving_Ctch%_prev', 'Receiving_1D_prev'
    ]

    if pos == 'QB':
        features = base_features + pass_features + rush_features
    else:
        features = base_features + rush_features + receiving_features

    X = df[features]
    # If a player scored 0 fantasy points in a season, I want that to be reflected in my model rather
    # than ignored as a NaN value.
    y = df['Fantasy_PPR'].fillna(0)
    # Set random_state to 42 so the split is reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.20,        
        random_state = 42,       
    )

    return X_train, X_test, y_train, y_test


def get_baseline_split(pos):
    """Create train/test split using ADP as the only independent
       variable.

    Args:
        pos: A string equal to 'QB', 'RB', 'WR', or 'TE' (case insensitve)
             representing the position of the players in the relevant dataset.

    Returns:
        Training and test datasets for the specified position containing ADP
        as the only independent variable.
    """
    df = pd.read_csv(f'{pos}_with_prev.csv')
    df = fix_0_values(df)
    # Use nested brackets to avoid a shape error when running the Random
    # Forest model (not needed for XGBoost model because it handles reshaping
    # internally, but probably good to have nonetheless).
    X = df[['ADP']]
    y = df['Fantasy_PPR'].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.20,        
        random_state = 42,       
    )

    return X_train, X_test, y_train, y_test


def get_corr_matrix(df, pos):
    """Create a correlation matrix for the specified position.

    Args:
        pos: A string equal to 'QB', 'RB', 'WR', 'TE'.
        df: A pandas dataframe of the training data for the specified position.

    Returns:
        Null. Generates a correlation matrix from the dataframe.
    """
    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    # Hide the upper triangle to reduce clutter.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        cmap='vlag',
        mask=mask,
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 4},      
        ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    plt.title(f'{pos} Correlation Matrix')
    plt.tight_layout()
    plt.show()


def get_best_rf(X_train, X_test, y_train, y_test):
    """Perform a Random Forest Grid Search using the provided data.
    
    Args:
        X_train: Dataframe of training observations for independent variables.
        X_test: Dataframe of test observations for independent variables.
        y_train: Dataframe of training observations for dependent variable.
        y_test: Dataframe of test observations for dependent variable.

    Returns:
        Null. Runs GridSearchCV on a base RandomForestRegressor using the provided data
        and prints the best MAE and best parameters, then uses those parameters to create
        and run a Best Estimator and prints the OOB Score, test MAE, and test R² score for
        the Best Estimator.
    """
    # I need to use TimeSeriesSplit to perform cross validation rather than just doing standard
    # cross validation because my data is time-ordered and I don't want my model to use future 
    # data to predict past events.
    tscv = TimeSeriesSplit(
        n_splits=5,      
        test_size=None   
    )

    param_grid = {
        'n_estimators':      [400, 800, 1200],
        # Although I used squared_error for training my initial model above, given that I'm testing
        # different parameters I want to test both criteria here as well.
        'criterion':         ['squared_error', 'absolute_error'],
        'max_depth':         [None, 8, 12, 16, 20],
        'min_samples_leaf':  [1, 2, 4, 8],
        'max_features':      [1.0, 'sqrt', 'log2'],
        'min_samples_split': [2, 4, 8, 12]
    }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    rf_base = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,        
        bootstrap=True,
        oob_score=True,   
    )

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring=mae_scorer,   
        cv=tscv,              
        n_jobs=-1,            
        verbose=2
    )

    grid.fit(X_train, y_train)

    print('Best MAE (CV):', -grid.best_score_)      
    print('Best params :')
    for k, v in grid.best_params_.items():
        print(f'  {k}: {v}')

    best_rf = grid.best_estimator_
    print('OOB R² :', best_rf.oob_score_)  

    y_pred = best_rf.predict(X_test)
    print('Test MAE:', mean_absolute_error(y_test, y_pred))
    print('Test R² :', r2_score(y_test, y_pred))

    importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)\
                .sort_values(ascending=False)
    print(importances.head(10))


def get_best_xgb(X_train, X_test, y_train, y_test):
    """Perform an XGBoost Grid Search using the provided data.
    
    Args:
        X_train: Dataframe of training observations for independent variables.
        X_test: Dataframe of test observations for independent variables.
        y_train: Dataframe of training observations for dependent variable.
        y_test: Dataframe of test observations for dependent variable.

    Returns:
        Null. Runs GridSearchCV on a base XGBRegressor using the provided data and 
        prints the best MAE and best parameters, then uses those parameters to create
        and run a Best Estimator and prints the OOB Score, test MAE, and test R² score for
        the Best Estimator.
    """
    tscv = TimeSeriesSplit(
        n_splits=5,      
        test_size=None   
    )

    param_grid = {
        'n_estimators':      [400, 800, 1200],   
        'learning_rate':     [0.1, 0.05, 0.02], 
        'max_depth':         [4, 6, 8],         
        'subsample':         [0.7, 0.9, 1.0],   
        'colsample_bytree':  [0.7, 0.9, 1.0],   
        'gamma':             [0, 1, 5],         
        'min_child_weight':  [1, 5, 10],        
        'reg_lambda':        [0.0, 1.0, 5.0],   
    }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    xgb_base = XGBRegressor(
        objective='reg:absoluteerror',     
        tree_method='hist',  # This setting improves the model's speed.              
        random_state=42,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=mae_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    print("Best MAE (CV):", -grid.best_score_)
    print("Best params :")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

    best_xgb = grid.best_estimator_

    y_pred = best_xgb.predict(X_test)
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test R² :", r2_score(y_test, y_pred))

    importances = pd.Series(best_xgb.feature_importances_, index=X_train.columns)\
                .sort_values(ascending=False)
    print(importances.head(10))


if __name__ == "__main__":
    # Replace position as applicable to test other positions.
    position_to_test = 'RB'
    # Replace with get_baseline_split to measure the baseline.
    X_train, X_test, y_train, y_test = get_train_test_split(position_to_test)
    get_corr_matrix(X_train, position_to_test)
    # Replace with get_best_rf to run Random Forest model.
    get_best_xgb(X_train, X_test, y_train, y_test)
