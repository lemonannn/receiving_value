import pandas as pd
import nfl_data_py as nfl
from pygam import LinearGAM, s, f, te


power_four_schools = [
    # SEC
    "Alabama", "Arkansas", "Auburn", "Florida", "Georgia",
    "Kentucky", "LSU", "Ole Miss", "Mississippi State",
    "Missouri", "South Carolina", "Tennessee", "Texas A&M",
    "Texas", "Vanderbilt", "Oklahoma",
    
    # Big Ten
    "Illinois", "Indiana", "Iowa", "Maryland", "Michigan",
    "Michigan State", "Minnesota", "Nebraska", "Northwestern",
    "Ohio State", "Penn State", "Purdue", "Rutgers", "UCLA",
    "USC", "Oregon", "Washington", "Wisconsin", 'Michigan St.',
    
    # Big 12
    "Arizona State", "Arizona", "Baylor", "BYU", "UCF",
    "Cincinnati", "Colorado", "Houston", "Iowa State", 
    "Kansas", "Kansas State", "Oklahoma State", "TCU",
    "Texas Tech", "Utah", "West Virginia", "Arizona St", "Oklahoma St.",
    
    # ACC
    "Boston College", "California", "Clemson", "Duke", "Florida State",
    "Georgia Tech", "Louisville", "Miami", "NC State",
    "North Carolina", "Pittsburgh", "Syracuse", "Stanford",
    "Virginia", "Virginia Tech", "Wake Forest", "SMU", 'Boston Col.', 'Florida St.',
    
    # Others
    "Notre Dame", "Washington State", "Oregon State", 'Oregon St.'
]

def get_year_table(year):
    receiving_summary = pd.read_csv(str(year) + '_recieving/receiving_summary.csv')
    receiving_depth = pd.read_csv(str(year) + '_recieving/receiving_depth.csv')
    receiving_scheme = pd.read_csv(str(year) + '_recieving/receiving_scheme.csv')
    receiving_concept = pd.read_csv(str(year) + '_recieving/receiving_concept.csv')
    
    receiving_summary = receiving_summary[['player', 'player_id', 'position', 'team_name', 'player_game_count',
                                                 'contested_receptions', 'contested_catch_rate',
                                                 'targets', 'yards', 'touchdowns', 'avg_depth_of_target',
                                                 'drop_rate', 'wide_rate', 'yprr']]

    receiving_depth = receiving_depth[['player_id', 'behind_los_yards', 'short_yards', 'medium_yards', 'deep_yards', 'behind_los_yprr', 'behind_los_receptions', 'short_yprr', 'short_receptions',
               'medium_yprr', 'medium_receptions', 'deep_yprr', 'deep_receptions']]

    receiving_scheme = receiving_scheme[['player_id', 'man_targets', 'man_yprr', 'man_avg_depth_of_target', 'man_yards', 
                                                'zone_targets', 'zone_yprr', 'zone_avg_depth_of_target', 'zone_yards']]
    
    recieving = pd.merge(receiving_summary, receiving_depth, on="player_id")
    recieving = pd.merge(recieving, receiving_scheme, on="player_id")
    
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
    recieving_full['man_yards_per_game'] = recieving_full['man_yards'] / recieving_full['player_game_count']
    recieving_full['zone_yards_per_game'] = recieving_full['zone_yards'] / recieving_full['player_game_count']
    
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
    
    