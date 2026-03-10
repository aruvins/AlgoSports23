import pandas as pd
import numpy as np

def run_rpl_model():
    # 1. Load data
    train = pd.read_csv('algosports23-predictions-2025/Train.csv')
    test = pd.read_csv('algosports23-predictions-2025/Predictions.csv')

    # --- NEW FEATURE: MARGIN CAPPING ---
    # Capping prevents a single massive blowout from distorting a team's power rating.
    # 50 is a common threshold in sports like this, but you can adjust based on data.
    MARGIN_CAP = 50 
    train['CappedMargin'] = train['HomeWinMargin'].clip(-MARGIN_CAP, MARGIN_CAP)
    
    # 2. Calculate Home Field Advantage (HFA) using capped margins
    hfa = train['CappedMargin'].mean()
    print(f"Calculated Home Field Advantage (Capped): {hfa:.2f} points")

    # --- NEW FEATURE: CONFERENCE STRENGTH ADJUSTMENT ---
    # We calculate how much a conference wins/loses when playing OTHER conferences.
    inter_conf = train[train['HomeConf'] != train['AwayConf']].copy()
    
    home_conf_perf = inter_conf.groupby('HomeConf')['CappedMargin'].mean()
    away_conf_perf = inter_conf.groupby('AwayConf')['CappedMargin'].mean() * -1
    
    # Combine to get a "Conference Strength" multiplier
    conf_strength = pd.concat([home_conf_perf, away_conf_perf]).groupby(level=0).mean().to_dict()
    
    # 3. Prepare game data
    games = []
    for _, row in train.iterrows():
        # Get the conference strength adjustments
        h_conf_adj = conf_strength.get(row['HomeConf'], 0)
        a_conf_adj = conf_strength.get(row['AwayConf'], 0)
        
        # Home perspective: Result - HFA + (Opponent Conf Strength - Own Conf Strength)
        games.append({
            'TeamID': row['HomeID'], 
            'OppID': row['AwayID'], 
            'Margin': row['CappedMargin'] - hfa + (a_conf_adj - h_conf_adj)
        })
        # Away perspective
        games.append({
            'TeamID': row['AwayID'], 
            'OppID': row['HomeID'], 
            'Margin': -row['CappedMargin'] + hfa + (h_conf_adj - a_conf_adj)
        })

    df_games = pd.DataFrame(games)
    all_team_ids = df_games['TeamID'].unique()

    # 4. Iterative SRS Calculation
    ratings = df_games.groupby('TeamID')['Margin'].mean().to_dict()
    for i in range(50):
        new_ratings = {}
        for team_id in all_team_ids:
            team_results = df_games[df_games['TeamID'] == team_id]
            avg_margin = team_results['Margin'].mean()
            opp_ratings = [ratings.get(opp, 0) for opp in team_results['OppID']]
            avg_sos = np.mean(opp_ratings)
            new_ratings[team_id] = avg_margin + avg_sos
        ratings = new_ratings

    # 5. Task 1: Create Rankings.xlsx
    team_names = pd.concat([
        train[['HomeID', 'HomeTeam']].rename(columns={'HomeID':'TeamID', 'HomeTeam':'Team'}),
        train[['AwayID', 'AwayTeam']].rename(columns={'AwayID':'TeamID', 'AwayTeam':'Team'})
    ]).drop_duplicates()

    rank_df = pd.DataFrame(list(ratings.items()), columns=['TeamID', 'PowerRating'])
    rank_df = rank_df.merge(team_names, on='TeamID').sort_values(by='PowerRating', ascending=False)
    rank_df['Rank'] = range(1, len(rank_df) + 1)
    
    rank_df[['TeamID', 'Team', 'Rank']].to_excel('Rankings.xlsx', index=False)
    print("✅ Rankings.xlsx created with Conference Adjustments and Capping.")

    # 6. Task 2: Predict Derby Margins
    def predict_margin(row):
        t1 = ratings.get(row['Team1_ID'], 0)
        t2 = ratings.get(row['Team2_ID'], 0)
        # We also apply conference adjustments to the final prediction
        conf1_adj = conf_strength.get(row['Team1_Conf'], 0)
        conf2_adj = conf_strength.get(row['Team2_Conf'], 0)
        
        return round((t1 + conf1_adj) - (t2 + conf2_adj), 2)

    test['Team1_WinMargin'] = test.apply(predict_margin, axis=1)
    test.to_csv('Predictions.csv', index=False)
    print("✅ Predictions.csv updated.")

    # ----------------- Optional: Model Evaluation -----------------
    # 7. Model Evaluation
    # --------------------------------------------------------------
    # Re-predict training games to see how well the ratings fit the history
    train['Pred'] = train.apply(lambda r: ratings[r['HomeID']] - ratings[r['AwayID']] + hfa, axis=1)
    acc = ((train['HomeWinMargin'] > 0) == (train['Pred'] > 0)).mean()
    mae = np.abs(train['HomeWinMargin'] - train['Pred']).mean()
    
    print(f"\n--- Model Performance ---")
    print(f"Win/Loss Accuracy: {acc:.1%}")
    print(f"Mean Absolute Error: {mae:.2f} points")

if __name__ == "__main__":
    run_rpl_model()