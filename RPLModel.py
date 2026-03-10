import pandas as pd
import numpy as np


############################
# CONFIG
############################

MARGIN_CAP = 60
ITERATIONS = 50
RECENCY_WEIGHT = True
USE_MASSEY = True
RIDGE_LAMBDA = 8


############################
# DATA LOADING
############################

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if 'Date' in train.columns:
        train['Date'] = pd.to_datetime(train['Date'])

    return train, test


############################
# FEATURE ENGINEERING
############################

def cap_margins(train):
    train['CappedMargin'] = train['HomeWinMargin'].clip(-MARGIN_CAP, MARGIN_CAP)
    return train


def compute_home_field_advantage(train):
    hfa = train['CappedMargin'].mean()
    print(f"Home Field Advantage: {hfa:.2f}")
    return hfa


def compute_recency_weights(train):
    if 'Date' not in train.columns:
        train['Weight'] = 1
        return train

    train['DayNum'] = (train['Date'] - train['Date'].min()).dt.days
    train['Weight'] = 0.5 + 0.5 * (train['DayNum'] / train['DayNum'].max())

    return train


############################
# CONFERENCE STRENGTH
############################

def compute_conference_strength(train):

    inter_conf = train[train['HomeConf'] != train['AwayConf']].copy()

    home_perf = inter_conf.groupby('HomeConf')['CappedMargin'].mean()
    away_perf = inter_conf.groupby('AwayConf')['CappedMargin'].mean() * -1

    conf_strength = pd.concat([home_perf, away_perf]).groupby(level=0).mean()

    return conf_strength.to_dict()


############################
# GAME MATRIX BUILDER
############################

def build_game_matrix(train, hfa, conf_strength):

    games = []

    for _, row in train.iterrows():

        h_adj = conf_strength.get(row['HomeConf'], 0)
        a_adj = conf_strength.get(row['AwayConf'], 0)

        games.append({
            'TeamID': row['HomeID'],
            'OppID': row['AwayID'],
            'Margin': row['CappedMargin'] - hfa + (a_adj - h_adj)
        })

        games.append({
            'TeamID': row['AwayID'],
            'OppID': row['HomeID'],
            'Margin': -row['CappedMargin'] + hfa + (h_adj - a_adj)
        })

    return pd.DataFrame(games)


############################
# SRS MODEL
############################

def compute_srs_ratings(df_games, iterations=ITERATIONS):

    teams = df_games['TeamID'].unique()
    ratings = df_games.groupby('TeamID')['Margin'].mean().to_dict()

    for _ in range(iterations):

        new_ratings = {}

        for team in teams:

            team_games = df_games[df_games['TeamID'] == team]

            avg_margin = team_games['Margin'].mean()

            opp_ratings = [
                ratings.get(opp, 0) for opp in team_games['OppID']
            ]

            avg_sos = np.mean(opp_ratings) if len(opp_ratings) > 0 else 0

            new_ratings[team] = avg_margin + avg_sos

        ratings = new_ratings

    return ratings


############################
# MASSEY MODEL
############################

def compute_massey_ratings(train):

    teams = pd.concat([train['HomeID'], train['AwayID']]).unique()

    team_index = {t: i for i, t in enumerate(teams)}

    n_teams = len(teams)
    n_games = len(train)

    M = np.zeros((n_games, n_teams))
    y = train['CappedMargin'].values

    for k, row in train.iterrows():

        i = team_index[row['HomeID']]
        j = team_index[row['AwayID']]

        M[k, i] = 1
        M[k, j] = -1

    ratings_vec = np.linalg.lstsq(M, y, rcond=None)[0]

    ratings = {team: ratings_vec[team_index[team]] for team in teams}

    return ratings

def compute_ridge_massey_ratings(train, ridge_lambda=8):

    teams = pd.concat([train['HomeID'], train['AwayID']]).unique()

    team_index = {t: i for i, t in enumerate(teams)}

    n_teams = len(teams)
    n_games = len(train)

    M = np.zeros((n_games, n_teams))
    y = train['CappedMargin'].values

    for k, row in train.iterrows():

        i = team_index[row['HomeID']]
        j = team_index[row['AwayID']]

        M[k, i] = 1
        M[k, j] = -1

    # Ridge solution
    A = M.T @ M + ridge_lambda * np.identity(n_teams)
    b = M.T @ y

    ratings_vec = np.linalg.solve(A, b)

    ratings = {team: ratings_vec[team_index[team]] for team in teams}

    return ratings


############################
# TEAM RANKINGS
############################

def generate_rankings(train, ratings):

    team_names = pd.concat([
        train[['HomeID', 'HomeTeam']].rename(columns={'HomeID': 'TeamID', 'HomeTeam': 'Team'}),
        train[['AwayID', 'AwayTeam']].rename(columns={'AwayID': 'TeamID', 'AwayTeam': 'Team'})
    ]).drop_duplicates()

    rank_df = pd.DataFrame(list(ratings.items()), columns=['TeamID', 'PowerRating'])

    rank_df = rank_df.merge(team_names, on='TeamID')

    rank_df = rank_df.sort_values(by='PowerRating', ascending=False)

    rank_df['Rank'] = range(1, len(rank_df) + 1)

    return rank_df


############################
# DERBY PREDICTIONS
############################

def predict_derby_margins(test, ratings, conf_strength):

    def predict(row):

        t1 = ratings.get(row['Team1_ID'], 0)
        t2 = ratings.get(row['Team2_ID'], 0)

        c1 = conf_strength.get(row['Team1_Conf'], 0)
        c2 = conf_strength.get(row['Team2_Conf'], 0)

        return round((t1 + c1) - (t2 + c2), 2)

    test['Team1_WinMargin'] = test.apply(predict, axis=1)

    return test


############################
# MODEL EVALUATION
############################

def evaluate_model(train, ratings, hfa):

    train['Pred'] = train.apply(
        lambda r: ratings[r['HomeID']] - ratings[r['AwayID']] + hfa,
        axis=1
    )

    acc = ((train['HomeWinMargin'] > 0) == (train['Pred'] > 0)).mean()

    mae = np.abs(train['HomeWinMargin'] - train['Pred']).mean()

    rmse = np.sqrt(np.mean((train['HomeWinMargin'] - train['Pred'])**2))

    print("\n--- Model Performance ---")
    print(f"Win/Loss Accuracy: {acc:.2%}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")


############################
# MAIN PIPELINE
############################

def run_model():

    train, test = load_data(
        "algosports23-predictions-2025/Train.csv",
        "algosports23-predictions-2025/Predictions.csv"
    )

    train = cap_margins(train)

    if RECENCY_WEIGHT:
        train = compute_recency_weights(train)

    hfa = compute_home_field_advantage(train)

    conf_strength = compute_conference_strength(train)

    if USE_MASSEY:
        ratings = compute_ridge_massey_ratings(train, RIDGE_LAMBDA)
    else:
        df_games = build_game_matrix(train, hfa, conf_strength)
        ratings = compute_srs_ratings(df_games)

    rankings = generate_rankings(train, ratings)

    predictions = predict_derby_margins(test, ratings, conf_strength)

    evaluate_model(train, ratings, hfa)

    rankings[['TeamID', 'Team', 'Rank']].to_excel("Rankings.xlsx", index=False)

    predictions.to_csv("Predictions.csv", index=False)

    print("\nFiles created:")
    print("Rankings.xlsx")
    print("Predictions.csv")


############################

if __name__ == "__main__":
    run_model()