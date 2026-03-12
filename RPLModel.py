import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

############################
# CONFIG
############################
MARGIN_CAP = 60  # Caps extreme margins to prevent outliers from dominating the model. Adjust based on data distribution.
RIDGE_LAMBDA = 2
BT_ITERATIONS = 1000
MASSEY_WEIGHT = 0.7  # Higher weight on Massey because LB is scored on RMSE (Margin)
BT_WEIGHT = 0.3      # Bradley-Terry helps stabilize the ranking for W/L logic

############################
# DATA LOADING
############################
def load_data():
    train = pd.read_csv("algosports23-predictions-2025/Train.csv")
    test = pd.read_csv("algosports23-predictions-2025/Predictions.csv")
    return train, test

############################
# FEATURE ENGINEERING & HFA
############################
def preprocess(train):
    train['CappedMargin'] = train['HomeWinMargin'].clip(-MARGIN_CAP, MARGIN_CAP)
    hfa = train['CappedMargin'].mean()
    print(f"Calculated Home Field Advantage: {hfa:.2f}")
    return train, hfa

def compute_conference_strength(train):
    # Calculate how conferences perform against other conferences
    inter_conf = train[train['HomeConf'] != train['AwayConf']].copy()
    home_perf = inter_conf.groupby('HomeConf')['CappedMargin'].mean()
    away_perf = inter_conf.groupby('AwayConf')['CappedMargin'].mean() * -1
    conf_strength = pd.concat([home_perf, away_perf]).groupby(level=0).mean()
    return conf_strength.to_dict()

############################
# MASSEY MODEL (For Margin/RMSE)
############################


def compute_ridge_massey(train, hfa, ridge_lambda):
    teams = pd.concat([train['HomeID'], train['AwayID']]).unique()
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams, n_games = len(teams), len(train)

    M = np.zeros((n_games, n_teams))
    # Target is Margin adjusted for HFA (Neutralized)
    y = train['CappedMargin'].values - hfa

    for k, row in train.iterrows():
        M[k, team_idx[row['HomeID']]] = 1
        M[k, team_idx[row['AwayID']]] = -1

    # Solve using Ridge Regression to handle small sample sizes per team
    A = M.T @ M + ridge_lambda * np.identity(n_teams)
    b = M.T @ y
    ratings_vec = np.linalg.solve(A, b)
    
    return {team: ratings_vec[team_idx[team]] for team in teams}

############################
# BRADLEY-TERRY MODEL (For Win Prob)
############################
def compute_bradley_terry(train, iterations=BT_ITERATIONS):
    teams = pd.concat([train['HomeID'], train['AwayID']]).unique()
    p = {team: 1.0 for team in teams}
    
    # Pre-calculate wins
    wins = train.apply(lambda x: x['HomeID'] if x['HomeWinMargin'] > 0 else x['AwayID'], axis=1).value_counts().to_dict()
    
    for _ in range(iterations):
        new_p = {}
        for i in teams:
            denom = 0
            games_i = train[(train['HomeID'] == i) | (train['AwayID'] == i)]
            for _, row in games_i.iterrows():
                j = row['AwayID'] if row['HomeID'] == i else row['HomeID']
                denom += 1.0 / (p[i] + p[j])
            new_p[i] = wins.get(i, 0) / denom if denom > 0 else 0.01
        
        # Normalize to prevent overflow
        avg_p = np.mean(list(new_p.values()))
        p = {k: v / avg_p for k, v in new_p.items()}
        
    return {k: np.log(v) for k, v in p.items()}

############################
# SCALING & ENSEMBLING
############################
def normalize(d):
    # Scales a dictionary of ratings to Mean=0, Std=1 for fair blending
    vals = list(d.values())
    m, s = np.mean(vals), np.std(vals)
    return {k: (v - m) / s for k, v in d.items()}

############################
# MAIN PIPELINE
############################
# def run1_model():
#     train, test = load_data()
#     train, hfa = preprocess(train)
#     conf_strength = compute_conference_strength(train)

#     # 1. Compute individual models
#     massey_raw = compute_ridge_massey(train, hfa, RIDGE_LAMBDA)
#     bt_raw = compute_bradley_terry(train)

#     # 2. Normalize and Blend for Rankings
#     m_norm = normalize(massey_raw)
#     bt_norm = normalize(bt_raw)
    
#     ensemble_ratings = {
#         t: (MASSEY_WEIGHT * m_norm[t]) + (BT_WEIGHT * bt_norm[t]) 
#         for t in m_norm.keys()
#     }

#     # 3. Predict Derby Margins (Strictly Neutral Site)
#     def predict(row):
#         # Use Massey + Conf Strength (best for Margin/RMSE)
#         t1_val = massey_raw.get(row['Team1_ID'], 0) + conf_strength.get(row['Team1_Conf'], 0)
#         t2_val = massey_raw.get(row['Team2_ID'], 0) + conf_strength.get(row['Team2_Conf'], 0)
#         return round(t1_val - t2_val, 2)

#     test['Team1_WinMargin'] = test.apply(predict, axis=1)

#     # 4. Generate Rankings.xlsx
#     team_names = pd.concat([
#         train[['HomeID', 'HomeTeam']].rename(columns={'HomeID': 'TeamID', 'HomeTeam': 'Team'}),
#         train[['AwayID', 'AwayTeam']].rename(columns={'AwayID': 'TeamID', 'AwayTeam': 'Team'})
#     ]).drop_duplicates()

#     rank_df = pd.DataFrame(list(ensemble_ratings.items()), columns=['TeamID', 'PowerRating'])
#     rank_df = rank_df.merge(team_names, on='TeamID').sort_values('PowerRating', ascending=False)
#     rank_df['Rank'] = range(1, len(rank_df) + 1)

#     # Export
#     rank_df[['Rank', 'TeamID', 'Team', 'PowerRating']].to_excel("Rankings.xlsx", index=False)
#     test.to_csv("Predictions.csv", index=False)
#     print("Files created: Rankings.xlsx and Predictions.csv")

def run_model(train_data, test_data, is_validation=True):
    """
    Trains the Massey Rating model and returns predictions.
    If is_validation is True, it adds HFA to the prediction (for regular season games).
    If False, it treats them as Neutral Site (for Derby games).
    """
    # Calculate HFA from training data
    hfa = train_data['HomeWinMargin'].mean()
    train_data = train_data.copy()
    train_data['CappedMargin'] = train_data['HomeWinMargin'].clip(-MARGIN_CAP, MARGIN_CAP)
    
    # Calculate Conference Strength
    inter_conf = train_data[train_data['HomeConf'] != train_data['AwayConf']].copy()
    all_confs = pd.concat([train_data['HomeConf'], train_data['AwayConf']]).unique()
    
    conf_strength = {}
    for conf in all_confs:
        h_perf = inter_conf[inter_conf['HomeConf'] == conf]['CappedMargin'].mean()
        a_perf = inter_conf[inter_conf['AwayConf'] == conf]['CappedMargin'].mean()
        conf_strength[conf] = np.nanmean([h_perf, -a_perf]) if not (np.isnan(h_perf) and np.isnan(a_perf)) else 0

    # Ridge Massey Ratings
    teams = pd.concat([train_data['HomeID'], train_data['AwayID']]).unique()
    team_to_idx = {t: i for i, t in enumerate(teams)}
    idx_to_team = {i: t for i, t in enumerate(teams)}
    n_teams = len(teams)
    
    X = np.zeros((len(train_data), n_teams))
    y = train_data['CappedMargin'].values - hfa

    for k, (_, row) in enumerate(train_data.iterrows()):
        X[k, team_to_idx[row['HomeID']]] = 1
        X[k, team_to_idx[row['AwayID']]] = -1

    A = X.T @ X + RIDGE_LAMBDA * np.eye(n_teams)
    b = X.T @ y
    r = np.linalg.solve(A, b)
    ratings = {idx_to_team[i]: r[i] for i in range(n_teams)}

    # Predictions
    preds = []
    for _, row in test_data.iterrows():
        # Handle different column names between Train and Predictions CSVs
        t1_id = row['HomeID'] if 'HomeID' in row else row['Team1_ID']
        t2_id = row['AwayID'] if 'AwayID' in row else row['Team2_ID']
        t1_conf = row['HomeConf'] if 'HomeConf' in row else row['Team1_Conf']
        t2_conf = row['AwayConf'] if 'AwayConf' in row else row['Team2_Conf']
        
        val1 = ratings.get(t1_id, 0) + conf_strength.get(t1_conf, 0)
        val2 = ratings.get(t2_id, 0) + conf_strength.get(t2_conf, 0)
        
        prediction = (val1 - val2)
        if is_validation:
            prediction += hfa # Add HFA back for regular season validation games
            
        preds.append(round(prediction, 2))
        
    return preds, ratings

if __name__ == "__main__":
    # run1_model()


    # --- STEP 1: LOCAL VALIDATION ---
    train, test = load_data()
    train_df = train.copy()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse = []


    print(f"{'Fold':<10} | {'RMSE':<10}")
    print("-" * 22)

    for i, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        t_fold, v_fold = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        # Run model on this specific fold
        preds, _ = run_model(t_fold, v_fold, is_validation=True)
        
        # Calculate error for this specific fold
        rmse = np.sqrt(mean_squared_error(v_fold['HomeWinMargin'], preds))
        cv_rmse.append(rmse)
        
        print(f"Fold {i+1:<5} | {rmse:<10.4f}")

    print("-" * 22)
    print(f"{'AVERAGE':<10} | {np.mean(cv_rmse):<10.4f}")
    print(f"{'STD DEV':<10} | {np.std(cv_rmse):<10.4f}")

    # --- STEP 2: FINAL SUBMISSION GENERATION ---
    # Now use 100% of the training data to predict the actual 75 Derby games
    test_df = test.copy()
    final_preds, final_ratings = run_model(train_df, test_df, is_validation=False)

    test_df['Team1_WinMargin'] = final_preds
    test_df.to_csv('Predictions.csv', index=False)

    # Generate Rankings.xlsx
    team_names = pd.concat([
        train_df[['HomeID', 'HomeTeam']].rename(columns={'HomeID': 'TeamID', 'HomeTeam': 'Team'}),
        train_df[['AwayID', 'AwayTeam']].rename(columns={'AwayID': 'TeamID', 'AwayTeam': 'Team'})
    ]).drop_duplicates()

    rank_df = pd.DataFrame(list(final_ratings.items()), columns=['TeamID', 'PowerRating'])
    rank_df = rank_df.merge(team_names, on='TeamID').sort_values('PowerRating', ascending=False)
    rank_df['Rank'] = range(1, len(rank_df) + 1)
    rank_df[['Rank', 'TeamID', 'Team', 'PowerRating']].to_excel('Rankings.xlsx', index=False)

    print("\nSuccess! Your 'Local Score' is ready. Submission files updated.")