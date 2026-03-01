import pandas as pd
import nfl_data_py as nfl
from pygam import LinearGAM, s, f, te


power_four_schools = [
    # SEC
    "Alabama", "Arkansas", "Auburn", "Florida", "Georgia",
    "Kentucky", "LSU", "Ole Miss", "Mississippi State",
    "Missouri", "South Carolina", "Tennessee", "Texas A&M",
    "Texas", "Vanderbilt", "Oklahoma", 'S CAROLINA', 'MISS STATE',
    
    # Big Ten
    "Illinois", "Indiana", "Iowa", "Maryland", "Michigan",
    "Michigan State", "Minnesota", "Nebraska", "Northwestern",
    "Ohio State", "Penn State", "Purdue", "Rutgers", "UCLA",
    "USC", "Oregon", "Washington", "Wisconsin", 'Michigan St.',
    'MICH STATE',
    
    # Big 12
    "Arizona State", "Arizona", "Baylor", "BYU", "UCF",
    "Cincinnati", "Colorado", "Houston", "Iowa State", 
    "Kansas", "Kansas State", "Oklahoma State", "TCU",
    "Texas Tech", "Utah", "West Virginia", "Arizona St", "Oklahoma St.",
    'Kansas St.', 'OKLA STATE', 'W VIRGINIA',
    
    # ACC
    "Boston College", "California", "Clemson", "Duke", "Florida State",
    "Georgia Tech", "Louisville", "Miami", "NC State",
    "North Carolina", "Pittsburgh", "Syracuse", "Stanford",
    "Virginia", "Virginia Tech", "Wake Forest", "SMU", 'Boston Col.', 'Florida St.',
    'Miami (FL)', 'WAKE', 'Florida St', 'N CAROLINA', 'MIAMI FL', 'CAL', 'GA TECH',
    'VA TECH', 'BOSTON COL',
    
    # Others
    "Notre Dame", "Washington State", "Oregon State", 'Oregon St.', 'Washington St.', 
    'OREGON ST', 'Washington St', 'WASH STATE'
]

def get_year_table(year):
    receiving_summary = pd.read_csv(str(year) + '_recieving/receiving_summary.csv')
    receiving_depth = pd.read_csv(str(year) + '_recieving/receiving_depth.csv')
    ## receiving_scheme = pd.read_csv(str(year) + '_recieving/receiving_scheme.csv')
    receiving_concept = pd.read_csv(str(year) + '_recieving/receiving_concept.csv')
    
    receiving_summary = receiving_summary[['player', 'player_id', 'position', 'team_name', 'player_game_count',
                                                 'contested_receptions', 'contested_catch_rate',
                                                 'targets', 'yards', 'touchdowns', 'avg_depth_of_target',
                                                 'drop_rate', 'wide_rate', 'yprr']]

    receiving_depth = receiving_depth[['player_id', 'behind_los_yards', 'short_yards', 'medium_yards', 'deep_yards', 'behind_los_yprr', 'behind_los_receptions', 'short_yprr', 'short_receptions',
               'medium_yprr', 'medium_receptions', 'deep_yprr', 'deep_receptions']]

    # receiving_scheme = receiving_scheme[['player_id', 'man_targets', 'man_yprr', 'man_avg_depth_of_target', 'man_yards', 
    #                                             'zone_targets', 'zone_yprr', 'zone_avg_depth_of_target', 'zone_yards']]
    
    recieving = pd.merge(receiving_summary, receiving_depth, on="player_id")
    # recieving = pd.merge(recieving, receiving_scheme, on="player_id")
    
    recieving['player'] = recieving['player'].str.removesuffix(' III')
    recieving['player'] = recieving['player'].str.removesuffix(' Jr.')
    
    draft_picks = nfl.import_draft_picks()
    draft_picks = draft_picks[draft_picks['season'].isin([int(year)+1])]
    draft_picks = draft_picks[draft_picks['position'] == 'WR']
    draft_picks = draft_picks[['pick', 'pfr_player_name', 'college']]
    
    recieving_full = pd.merge(recieving, draft_picks, left_on='player', right_on='pfr_player_name')
    
    recieving_full['var_depth'] = recieving_full[['behind_los_yards', 'short_yards', 'medium_yards', 'deep_yards']].var(axis=1)
    
    recieving_full['perc_deep_yards'] = recieving_full['deep_yards'] / (recieving_full['deep_yards'] + 
                                                                        recieving_full['medium_yards'] +
                                                                        recieving_full['short_yards'] + 
                                                                        recieving_full['behind_los_yards'])
    
    recieving_full['yards_per_game'] = recieving_full['yards'] / recieving_full['player_game_count']
    # recieving_full['man_yards_per_game'] = recieving_full['man_yards'] / recieving_full['player_game_count']
    # recieving_full['zone_yards_per_game'] = recieving_full['zone_yards'] / recieving_full['player_game_count']
    
    recieving_full['is_power_four'] = (recieving_full['college'].str.lower().isin([school.lower() for school in power_four_schools]) |
                                       recieving_full['team_name'].str.lower().isin([school.lower() for school in power_four_schools]))


    return recieving_full

def merge_datasets(datasets):
    return pd.concat(datasets)

def create_gam(dataset):
    
    X = dataset[['man_yprr', 'man_yards', 'zone_yprr', 'zone_yards', 'deep_yards', 'medium_yards', 'short_yards', 
                         'behind_los_yards', 'var_depth', 'drop_rate', 'wide_rate', 'man_avg_depth_of_target', 
                         'zone_avg_depth_of_target', 'contested_receptions', 'contested_catch_rate', 'is_power_four', 
                         'perc_deep_yards', 'man_yards_per_game', 'zone_yards_per_game']]
    
    y = dataset['pick']
    
    gam = LinearGAM(s(0) + s(2) + s(17) + s(18)
                + f(15))
    gam.fit(X, y)
    
    gam.summary()
    
    predicted_slot = gam.predict(X)
    
    dataset['predicted_slot'] = predicted_slot
    
    dataset[['player', 'predicted_slot']]
    
def get_receiving_dataset(start_year, end_year):
    
    recieving_list = []
    
    for i in range(start_year, end_year+1):
        
        receiving_summary = pd.read_csv(str(i) + '_recieving/receiving_summary.csv')
        receiving_depth = pd.read_csv(str(i) + '_recieving/receiving_depth.csv')
        
        receiving_summary = receiving_summary[['player', 'player_id', 'position', 'team_name', 'player_game_count',
                                                'drop_rate', 'wide_rate', 'yprr', 'targets']]

        receiving_depth = receiving_depth[['player_id', 'behind_los_yards', 'short_yards', 'medium_yards', 'deep_yards']]
        
        recieving = pd.merge(receiving_summary, receiving_depth, on="player_id")
        
        recieving['year'] = i
        
        recieving_list.append(recieving)
        
    recieving = pd.concat(recieving_list, ignore_index=True)
    return recieving

def get_draft_picks(start_year, end_year):
    
    draft_picks = nfl.import_draft_picks(years=list(range(start_year, end_year+1)))
    draft_picks = draft_picks[draft_picks['position'] == 'WR']
    draft_picks = draft_picks[['pick', 'pfr_player_name', 'college']]
    return draft_picks


def add_columns(recieving_full):
    
    recieving_full['is_power_four'] = (recieving_full['team_name'].str.lower().isin([school.lower() for school in power_four_schools]))
    
    recieving_full['deep_ypg'] = recieving_full['deep_yards'] / recieving_full['player_game_count']
    recieving_full['medium_ypg'] = recieving_full['medium_yards'] / recieving_full['player_game_count']
    recieving_full['short_ypg'] = recieving_full['short_yards'] / recieving_full['player_game_count']
    recieving_full['behind_los_ypg'] = recieving_full['behind_los_yards'] / recieving_full['player_game_count']
    
    return recieving_full
    
def get_best_numbers(dataset):
    
    dataset = dataset[dataset['targets'] > 30]
    
    # gets rid of non-power four years if most recent season was power four
    latest = dataset.loc[dataset.groupby("player")["year"].idxmax()]
    latest_false = latest.loc[latest["is_power_four"] == False, "player"]
    dataset = dataset[(~dataset["player"].isin(latest_false)) | (dataset["is_power_four"] == False)]
        
    # takes max values in ypg and yprr, averages drop rate and wide rate
    # this is a bad way to do it, doesn't account for different usage
    # dataset = dataset.sort_values(["player", "year"])
    # filtered_dataset = (
    #     dataset
    #     .groupby("player")
    #     .agg({
    #         "behind_los_ypg": "max",
    #         "short_ypg": "max",
    #         "medium_ypg": "max",
    #         "deep_ypg": "max",
    #         "yprr": "max",
    #         "drop_rate": "mean",
    #         "wide_rate": "mean",
    #         "is_power_four": "last",
    #         "college": "last",
    #         "pick": "last",
    #     })
    #     .reset_index()
    # )
    
    return dataset
    
def get_dataset():
    
    receiving = get_receiving_dataset(2014, 2024)
    
    draft_picks = get_draft_picks(2015, 2025)
    
    recieving_full = pd.merge(receiving, draft_picks, left_on='player', right_on='pfr_player_name')
    
    receiving_plus = add_columns(recieving_full)
    
    receiving_filtered = get_best_numbers(receiving_plus)
    
    return receiving_filtered

def get_2026_prospects():
    
    receiving = get_receiving_dataset(2023, 2025)
    
    receiving = add_columns(receiving)
    
    receiving = get_best_numbers(receiving)
    
    receiving = receiving[receiving['is_power_four'] == True]
    
    return receiving
    
    