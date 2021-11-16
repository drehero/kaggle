from config import *

from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf


# Helper function to unpack json found in daily data
def unpack_json(json_str):
    return np.nan if pd.isna(json_str) else pd.read_json(json_str)


def unpack_data(data, dfs=None, is_test=False):
    if is_test:
        data = data.copy()
        data["date"] = data.index
        data = data.reset_index(drop=True)
    unpacked_data = {}
    for df_name in dfs:
        date_nested_table = data[['date', df_name]]
        date_nested_table = date_nested_table[~pd.isna(date_nested_table[df_name])].reset_index(drop = True)
        daily_dfs_collection = []
        for date_index, date_row in date_nested_table.iterrows():
            daily_df = pd.read_json(date_row[df_name])
            daily_df['dailyDataDate'] = date_row['date']
            daily_dfs_collection = daily_dfs_collection + [daily_df]
        # Concatenate all daily dfs into single df for each row
        try:
            unnested_table = (pd.concat(daily_dfs_collection,
              ignore_index = True).
              # Set and reset index to move 'dailyDataDate' to front of df
              set_index('dailyDataDate').
              reset_index()
              )
            # Creates 1 pandas df per unnested df from daily data read in, with same name
            unpacked_data[df_name] = unnested_table
        except ValueError:
            unpacked_data[df_name] = pd.DataFrame()
    return unpacked_data


# relevant feature names
TARGET_FEATURE_NAMES = ["target1", "target2", "target3", "target4"]
TRAIN_FEATURE_NAMES = ["nextDayPlayerEngagement", "rosters", "playerTwitterFollowers",
                       "teamTwitterFollowers", "games", #"teamBoxScores",
                       "playerBoxScores", "transactions", "awards", "standings", "events"]

TEST_FEATURE_NAMES = TRAIN_FEATURE_NAMES.copy()
TEST_FEATURE_NAMES.remove("nextDayPlayerEngagement")

PLAYERS_FEATURE_NAMES = ['playerId', 'DOB', 'mlbDebutDate', 'birthCountry', 'heightInches',
                         'weight', 'primaryPositionName']
TEAMS_FEATURE_NAMES = ['id', 'leagueName', 'divisionName']
ROSTERS_FEATURE_NAMES = ["dailyDataDate", "playerId", "teamId", "status"]
GAMES_FEATURE_NAMES = ["dailyDataDate", "gamePk", "gameType", "gameTimeUTC", "detailedGameState",
                       "isTie", "gameNumber", "doubleHeader", "dayNight", "scheduledInnings",
                       "gamesInSeries", "seriesDescription", "homeId", "homeWins", "homeLosses",
                       "homeWinPct", "homeWinner", "homeScore", "awayId", "awayWins", "awayLosses",
                       "awayWinPct", "awayWinner", "awayScore"]
TEAM_BOX_SCORES_FEATURE_NAMES = ['dailyDataDate', 'teamId', 'flyOuts', 'groundOuts',
                                 'runsScored', 'doubles', 'triples', 'homeRuns',
                                 'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits',
                                 'hitByPitch', 'atBats', 'caughtStealing', 'stolenBases',
                                 'groundIntoDoublePlay', 'groundIntoTriplePlay', 'plateAppearances',
                                 'totalBases', 'rbi', 'leftOnBase', 'sacBunts', 'sacFlies',
                                 'catchersInterference', 'pickoffs', 'airOutsPitching',
                                 'groundOutsPitching', 'runsPitching', 'doublesPitching',
                                 'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
                                 'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
                                 'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
                                 'stolenBasesPitching', 'inningsPitched', 'earnedRuns',
                                 'battersFaced', 'outsPitching', 'hitBatsmen', 'balks',
                                 'wildPitches', 'pickoffsPitching', 'rbiPitching', 'inheritedRunners',
                                 'inheritedRunnersScored', 'catchersInterferencePitching',
                                 'sacBuntsPitching', 'sacFliesPitching']
PLAYER_BOX_SCORES_FEATURE_NAMES = ['dailyDataDate', 'playerId', 'positionCode',
                                   'positionType', 'battingOrder', 'gamesPlayedBatting',
                                   'flyOuts', 'groundOuts', 'runsScored', 'doubles', 'triples',
                                   'homeRuns', 'strikeOuts', 'baseOnBalls', 'intentionalWalks',
                                   'hits', 'hitByPitch', 'atBats', 'caughtStealing',
                                   'stolenBases', 'groundIntoDoublePlay', 'groundIntoTriplePlay',
                                   'plateAppearances', 'totalBases', 'rbi', 'leftOnBase', 'sacBunts',
                                   'sacFlies', 'catchersInterference', 'pickoffs', 'gamesPlayedPitching',
                                   'gamesStartedPitching', 'completeGamesPitching', 'shutoutsPitching',
                                   'winsPitching', 'lossesPitching', 'flyOutsPitching',
                                   'airOutsPitching', 'groundOutsPitching', 'runsPitching',
                                   'doublesPitching', 'triplesPitching', 'homeRunsPitching',
                                   'strikeOutsPitching', 'baseOnBallsPitching', 'intentionalWalksPitching',
                                   'hitsPitching', 'hitByPitchPitching', 'atBatsPitching',
                                   'caughtStealingPitching', 'stolenBasesPitching', 'inningsPitched',
                                   'saveOpportunities', 'earnedRuns', 'battersFaced', 'outsPitching',
                                   'pitchesThrown', 'balls', 'strikes', 'hitBatsmen', 'balks',
                                   'wildPitches', 'pickoffsPitching', 'rbiPitching', 'gamesFinishedPitching',
                                   'inheritedRunners', 'inheritedRunnersScored',
                                   'catchersInterferencePitching', 'sacBuntsPitching', 'sacFliesPitching',
                                   'saves', 'holds', 'blownSaves', 'assists', 'putOuts', 'errors', 'chances']
TRANSACTIONS_FEATURE_NAMES = ['dailyDataDate', 'transactionId', 'playerId', 'date', 'fromTeamId',
                              'toTeamId', 'effectiveDate', 'resolutionDate', 'typeDesc']
TRANSACTION_DESC_COLS = ['assigned', 'claimedOffWaivers', 'declaredFreeAgency', 'designatedforAssignment',
                         'numberChange', 'optioned', 'outrighted', 'recalled', 'released', 'retired',
                         'returned', 'selected', 'signed', 'signedasFreeAgent', 'statusChange',
                         'trade', 'hasTransactions']
AWARDS_FEATURE_NAMES = ["dailyDataDate", "awardId", "awardName", "playerId"]
STANDINGS_FEATURE_NAMES = ['dailyDataDate', 'teamId', 'streakCode', 'divisionRank', 'leagueRank',
                           'wildCardRank', 'leagueGamesBack', 'sportGamesBack', 'divisionGamesBack',
                           'wins', 'losses', 'pct', 'runsAllowed', 'runsScored', 'divisionChamp',
                           'divisionLeader', 'wildCardLeader', 'eliminationNumber',
                           'wildCardEliminationNumber', 'homeWins', 'homeLosses', 'awayWins',
                           'awayLosses', 'lastTenWins', 'lastTenLosses', 'extraInningWins',
                           'extraInningLosses', 'oneRunWins', 'oneRunLosses', 'dayWins',
                           'dayLosses', 'nightWins', 'nightLosses', 'grassWins', 'grassLosses',
                           'turfWins', 'turfLosses', 'divWins', 'divLosses', 'xWinLossPct']
MLB_SOCIAL_MEDIA_FEATURE_NAMES = ['id', 't_tweets',
                                  'i_posts', 'i_followers', 'i_av_likes', 'i_av_comments',
                                  'y_subscribers', 'y_uploads',
                                  'f_follow',
                                  'tik_followers', 'tik_likes',
                                  'p_followers', 'p_following']



NUMERIC_FEATURE_NAMES = [# twitter
                         "numberOfFollowersPlayer", "numberOfFollowersTeam",
                         # players
                         "playerAge", "daysSinceMlbDebute", "heightInches", "weight",
                         # games
                         "gameNumber", "scheduledInnings", "gamesInSeries", "teamWins",
                         "teamLosses", "teamWinPct", "teamScore", "oppWins", "oppLosses",
                         "oppWinPct", "oppScore",
                         # team box scores
                         #'flyOutsTeam', 'groundOutsTeam', 'runsScoredTeam', 'doublesTeam',
                         #'triplesTeam', 'homeRunsTeam', 'strikeOutsTeam', 'baseOnBallsTeam', 
                         #'intentionalWalksTeam', 'hitsTeam','hitByPitchTeam', 'atBatsTeam',
                         #'caughtStealingTeam', 'stolenBasesTeam', 'groundIntoDoublePlayTeam',
                         #'groundIntoTriplePlayTeam', 'plateAppearancesTeam', 'totalBasesTeam',
                         #'rbiTeam', 'leftOnBaseTeam', 'sacBuntsTeam', 'sacFliesTeam',
                         #'catchersInterferenceTeam', 'pickoffsTeam', 'airOutsPitchingTeam',
                         #'groundOutsPitchingTeam', 'runsPitchingTeam', 'doublesPitchingTeam',
                         #'triplesPitchingTeam', 'homeRunsPitchingTeam', 'strikeOutsPitchingTeam',
                         #'baseOnBallsPitchingTeam', 'intentionalWalksPitchingTeam', 'hitsPitchingTeam',
                         #'hitByPitchPitchingTeam', 'atBatsPitchingTeam', 'caughtStealingPitchingTeam',
                         #'stolenBasesPitchingTeam', 'inningsPitchedTeam', 'earnedRunsTeam',
                         #'battersFacedTeam', 'outsPitchingTeam', 'hitBatsmenTeam', 'balksTeam',
                         #'wildPitchesTeam', 'pickoffsPitchingTeam', 'rbiPitchingTeam',
                         #'inheritedRunnersTeam', 'inheritedRunnersScoredTeam',
                         #'catchersInterferencePitchingTeam', 'sacBuntsPitchingTeam',
                         #'sacFliesPitchingTeam',
                         # player box scores
                         'gamesPlayedBatting', 'flyOuts', 'groundOuts', 'runsScored',
                         'doubles', 'triples', 'homeRuns', 'strikeOuts', 'baseOnBalls',
                         'intentionalWalks', 'hits', 'hitByPitch', 'atBats', 'caughtStealing',
                         'stolenBases', 'groundIntoDoublePlay', 'groundIntoTriplePlay',
                         'plateAppearances', 'totalBases', 'rbi', 'leftOnBase', 'sacBunts',
                         'sacFlies', 'catchersInterference', 'pickoffs', 'gamesPlayedPitching',
                         'gamesStartedPitching', 'completeGamesPitching', 'shutoutsPitching',
                         'winsPitching', 'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
                         'groundOutsPitching', 'runsPitching', 'doublesPitching',
                         'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
                         'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
                         'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
                         'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
                         'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
                         'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
                         'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
                         'inheritedRunnersScored', 'catchersInterferencePitching',
                         'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
                         'assists', 'putOuts', 'errors', 'chances', 'inningsPitchedAsFrac',
                         'pitchingGameScore', 'battingOrderSpot', 'battingOrderSequence',
                         # transactions
                         'numTransactionInfos', 'effectiveDateDelta', 'resolutionDateDelta',
                         # awards
                         "numAwards",
                         # events
                         'pitches100mph', 'walkoffRBIAllowed', 'HRDist450ft', 'gameTyingRBI',
                         'goAheadRBI', 'walkoffRBI',
                         # standings
                         'wins', 'losses', 'pct', 'runsAllowed', 'runsScoredStandings',
                         'homeWins', 'homeLosses', 'awayWins', 'awayLosses', 'lastTenWins',
                         'lastTenLosses', 'extraInningWins', 'extraInningLosses', 'oneRunWins',
                         'oneRunLosses', 'dayWins', 'dayLosses', 'nightWins', 'nightLosses',
                         'grassWins', 'grassLosses', 'turfWins', 'turfLosses', 'divWins',
                         'divLosses', 'xWinLossPct', 'winStreak', 'loosingStreak',
                         # player target stats
                         'target1_mean', 'target1_std', 'target1_min', 'target1_25',
                         'target1_50', 'target1_75',
                         'target1_max', 'target2_mean',
                         'target2_std', 'target2_min', 'target2_25',
                         'target2_50', 'target2_75',
                         'target2_max', 'target3_mean', 'target3_std',
                         'target3_min', 'target3_25',
                         'target3_50', 'target3_75',
                         'target3_max', 'target4_mean', 'target4_std', 'target4_min', 'target4_25',
                         'target4_50', 'target4_75',
                         'target4_max',
                         # mlb social media
                         't_tweets', 'i_posts', 'i_followers',
                         'i_av_likes', 'i_av_comments', 'y_subscribers',
                         'y_uploads', 'f_follow', 'tik_followers', 'tik_likes',
                         'p_followers',
                         # player salary 
                         "salary"
                         ]
# lags
for lag in range(1, N_LAGS+1):
    NUMERIC_FEATURE_NAMES += [f"target1Lag{lag}", f"target2Lag{lag}", f"target3Lag{lag}", f"target4Lag{lag}"]
BINARY_FEATURE_NAMES = ["inSeason", "hasRoster", "daysSinceMlbDebuteWasMissing",
                        "isTie", "teamWinner", "oppWinner", "isHome", "hasGame",
                        "isNight", #"hasTeamBoxScores",
                        "hasPlayerBoxScores", "noHitter",
                        # transactions
                        'assigned', 'claimedOffWaivers', 'declaredFreeAgency',
                        'designatedforAssignment', 'numberChange', 'optioned', 'outrighted',
                        'recalled', 'released', 'retired', 'returned', 'selected', 'signed',
                        'signedasFreeAgent', 'statusChange', 'trade', 'hasTransactions',
                        # awards
                        "hasAwards",
                        # events
                        'hasEvents',
                        # standings
                        "divisionLeader", "divisionChamp", "hasStandings", "xWinLossPctWasMissing",
                        # social media
                        "hasSocialMediaData",
                        # salary
                        "salaryWasMissing", "hasSalary"
                       ]
CATEGORICAL_FEATURE_NAMES = ['playerId', 'day', 'week', 'month', 'year', 'dayOfWeek', 'dayOfYear',
                             'seasonPart', 'teamId', 'rosterStatus', 'birthCountry', 'primaryPositionName',
                             'leagueName', 'divisionName', 'gameType', 'detailedGameState',
                             'doubleHeader', 'seriesDescription', 'oppId', 'gameHour',
                             'positionCode', 'positionType', 'fromTeamId', 'toTeamId',
                             "awardCategory", "wildCardLeader", "divisionRank", "leagueRank",
                             "wildCardRank", "eliminationNumber", "wildCardEliminationNumber", 
                             'leagueGamesBack', 'sportGamesBack', 'divisionGamesBack']

#CATEGORICAL_FEATURE_NAMES += BINARY_FEATURE_NAMES
#BINARY_FEATURE_NAMES = []

FEATURE_NAMES = CATEGORICAL_FEATURE_NAMES + NUMERIC_FEATURE_NAMES + BINARY_FEATURE_NAMES

COLUMN_DEFAULTS = {feature_name: "NA" if feature_name in CATEGORICAL_FEATURE_NAMES else 0.0 for feature_name in FEATURE_NAMES}


# award categories to summarize some awards
AWARD_CATEGORIES = {
    "MiLB.com Organization All-Star": "milbAllStar",
    "FSL Mid-Season All-Star": "midSeasonAllStar",
    "MID Mid-Season All-Star": "midSeasonAllStar",
    "SOU Mid-Season All-Star": "midSeasonAllStar",
    "TEX Mid-Season All-Star": "midSeasonAllStar",
    "EAS Mid-Season All-Star": "midSeasonAllStar",
    "MEX Mid-Season All-Star": "midSeasonAllStar",
    "SAL Mid-Season All-Star": "midSeasonAllStar",
    "CAR Mid-Season All-Star": "midSeasonAllStar",
    "CAL Mid-Season All-Star": "midSeasonAllStar",
    "AFL Rising Stars": "risingStar",
    "Futures Game Selection": "futuresGameSelection",
    "ALPB Mid-Season All-Star": "midSeasonAllStar",
    "DSL Mid-Season All-Star": "midSeasonAllStar",
    "INT Mid-Season All-Star": "midSeasonAllStar",
    "World Series Championship": "worldSeriesChampionship",
    "AL All-Star": "LeagueAllStar",
    "NL All-Star": "LeagueAllStar",
    "PCL Mid-Season All-Star": "midSeasonAllStar",
    "AL Player of the Week": "LeaguePlayerOfTheWeek",
    "NL Player of the Week": "LeaguePlayerOfTheWeek",
    "NWL Mid-Season All-Star": "midSeasonAllStar",
    "PIO Mid-Season All-Star": "midSeasonAllStar",
    "NYP Mid-Season All-Star": "midSeasonAllStar",
    "Baseball America Major League All-Rookie Team": "baseballAmericaAward",
    "TEX Pitcher of the Week": "pitcherOfTheWeek",
    "CAL Pitcher of the Week": "pitcherOfTheWeek",
    "FSL Pitcher of the Week": "pitcherOfTheWeek",
    "INT Player of the Week": "playerOfTheWeek",
    "SAL Player of the Week": "playerOfTheWeek",
    "CAL Player of the Week": "playerOfTheWeek",
    "CAR Pitcher of the Week": "pitcherOfTheWeek",
    "FSL Player of the Week": "playerOfTheWeek",
    "PCL Player of the Week": "playerOfTheWeek",
    "PCL Pitcher of the Week": "pitcherOfTheWeek",
    "INT Pitcher of the Week": "pitcherOfTheWeek",
    "TEX Player of the Week": "playerOfTheWeek",
    "CAR Player of the Week": "playerOfTheWeek",
    "MID Pitcher of the Week": "pitcherOfTheWeek",
    "MID Player of the Week": "playerOfTheWeek",
    "EAS Pitcher of the Week": "pitcherOfTheWeek",
    "SAL Pitcher of the Week": "pitcherOfTheWeek",
    "EAS Player of the Week": "playerOfTheWeek",
    "SOU Player of the Week": "playerOfTheWeek",
    "SOU Pitcher of the Week": "pitcherOfTheWeek",
    "FSL Post-Season All-Star": "postSeasonAllStar",
    "Baseball America Rookie All-Star": "baseballAmericaAward",
    "SOU Post-Season All-Star": "postSeasonAllStar",
    "Baseball America Triple-A All-Star": "baseballAmericaAward",
    "Baseball America Low Class A All-Star": "baseballAmericaAward",
    "Baseball America Short-Season All-Star": "baseballAmericaAward",
    "Baseball America DSL All-Star": "baseballAmericaAward",
    "Baseball America Minor League All-Star": "baseballAmericaAward",
    "Baseball America Double-A All-Star": "baseballAmericaAward",
    "TEX Post-Season All-Star": "postSeasonAllStar",
    "APP Post-Season All-Star": "postSeasonAllStar",
    "SAL Post-Season All-Star": "postSeasonAllStar",
    "NWL Post-Season All-Star": "postSeasonAllStar",
    "CAL Post-Season All-Star": "postSeasonAllStar",
    "EAS Post-Season All-Star": "postSeasonAllStar",
    "AZL Post-Season All-Star": "postSeasonAllStar",
    "PIO Post-Season All-Star": "postSeasonAllStar",
    "CAR Post-Season All-Star": "postSeasonAllStar",
    "MID Post-Season All-Star": "postSeasonAllStar",
    "DSL Post-Season All-Star": "postSeasonAllStar",
    "INT Post-Season All-Star": "postSeasonAllStar",
    "PWL Post-Season All-Star": "postSeasonAllStar",
    "PCL Post-Season All-Star": "postSeasonAllStar",
    "GCL Post-Season All-Star": "postSeasonAllStar",
    "NWL Pitcher of the Week": "pitcherOfTheWeek",
    "PIO Player of the Week": "playerOfTheWeek",
    "NYP Player of the Week": "playerOfTheWeek",
    "APP Pitcher of the Week": "pitcherOfTheWeek",
    "NYP Pitcher of the Week": "pitcherOfTheWeek",
    "NWL Player of the Week": "playerOfTheWeek",
    "APP Player of the Week": "playerOfTheWeek",
    "PIO Pitcher of the Week": "pitcherOfTheWeek",
    "NL Player of the Month": "leaguePlayerOfTheMonth",
    "AL Reliever of the Month": "leaguePlayerOfTheMonth",
    "NL Rookie of the Month": "leaguePlayerOfTheMonth",
    "AL Player of the Month": "leaguePlayerOfTheMonth",
    "NL Pitcher of the Month": "leaguePlayerOfTheMonth",
    "AL Rookie of the Month": "leaguePlayerOfTheMonth",
    "AL Pitcher of the Month": "leaguePlayerOfTheMonth",
    "NL Reliever of the Month": "leaguePlayerOfTheMonth",
    "MiLB.com INT Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com CAR Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com MID Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com SOU Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com EAS Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com PCL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com CAL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com SAL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com FSL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com TEX Player of the Month": "milbPlayerOfTheMonth",
    "AFL Player of the Week": "playerOfTheWeek",
    "AFL Pitcher of the Week": "pitcherOfTheWeek",
    "MiLB.com NWL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com GCL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com AZL Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com NYP Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com APP Player of the Month": "milbPlayerOfTheMonth",
    "MiLB.com PIO Player of the Month": "milbPlayerOfTheMonth",
}


GAMES_TO_MERGE_COLUMNS = ['dailyDataDate', 'teamId', 'gamePk', 'gameType',
                          'detailedGameState', 'isTie', 'gameNumber', 'doubleHeader',
                          'scheduledInnings', 'gamesInSeries', 'seriesDescription',
                          'teamWins', 'teamLosses', 'teamWinPct', 'teamWinner', 'teamScore',
                          'oppId', 'oppWins', 'oppLosses', 'oppWinPct', 'oppWinner',
                          'oppScore', 'isHome', 'gameHour', 'isNight', 'hasGame']
PLAYER_BOX_SCORES_TO_MERGE_COLUMNS = ['dailyDataDate', 'playerId', 'positionCode',
                                      'positionType', 'gamesPlayedBatting', 'flyOuts',
                                      'groundOuts', 'runsScored', 'doubles', 'triples',
                                      'homeRuns', 'strikeOuts', 'baseOnBalls', 'intentionalWalks',
                                      'hits', 'hitByPitch', 'atBats', 'caughtStealing',
                                      'stolenBases', 'groundIntoDoublePlay', 'groundIntoTriplePlay',
                                      'plateAppearances', 'totalBases', 'rbi', 'leftOnBase',
                                      'sacBunts', 'sacFlies', 'catchersInterference',
                                      'pickoffs', 'gamesPlayedPitching',
                                      'gamesStartedPitching', 'completeGamesPitching',
                                      'shutoutsPitching', 'winsPitching', 'lossesPitching',
                                      'flyOutsPitching', 'airOutsPitching', 'groundOutsPitching',
                                      'runsPitching', 'doublesPitching', 'triplesPitching',
                                      'homeRunsPitching', 'strikeOutsPitching', 'baseOnBallsPitching',
                                      'intentionalWalksPitching', 'hitsPitching', 'hitByPitchPitching',
                                      'atBatsPitching', 'caughtStealingPitching', 'stolenBasesPitching',
                                      'inningsPitched', 'saveOpportunities', 'earnedRuns',
                                      'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
                                      'strikes', 'hitBatsmen', 'balks', 'wildPitches',
                                      'pickoffsPitching', 'rbiPitching', 'gamesFinishedPitching',
                                      'inheritedRunners', 'inheritedRunnersScored',
                                      'catchersInterferencePitching', 'sacBuntsPitching',
                                      'sacFliesPitching', 'saves', 'holds', 'blownSaves',
                                      'assists', 'putOuts', 'errors', 'chances',
                                      'inningsPitchedAsFrac', 'pitchingGameScore', 'noHitter',
                                      'battingOrderSpot', 'battingOrderSequence', 'hasPlayerBoxScores']
TRANSACTIONS_TO_MERGE_COLUMNS = ['dailyDataDate', 'playerId', 'fromTeamId', 'toTeamId',
                                 'numTransactionInfos', 'effectiveDateDelta', 'resolutionDateDelta',
                                 'assigned', 'claimedOffWaivers', 'declaredFreeAgency',
                                 'designatedforAssignment', 'numberChange', 'optioned', 'outrighted',
                                 'recalled', 'released', 'retired', 'returned', 'selected', 'signed',
                                 'signedasFreeAgent', 'statusChange', 'trade', 'hasTransactions']
AWARDS_TO_MERGE_COLUMNS = ['dailyDataDate', 'playerId', 'numAwards', 'awardCategory', 'hasAwards']
EVENTS_TO_MERGE_COLUMNS =  ['dailyDataDate', 'playerId', 'pitches100mph', 'walkoffRBIAllowed',
                            'HRDist450ft', 'gameTyingRBI', 'goAheadRBI', 'walkoffRBI', 'hasEvents']
STANDINGS_TO_MERGE_COLUMNS = ['dailyDataDate', 'teamId', 'divisionRank', 'leagueRank', 'wildCardRank',
                              'leagueGamesBack', 'sportGamesBack', 'divisionGamesBack', 'wins',
                              'losses', 'pct', 'runsAllowed', 'runsScoredStandings', 'divisionChamp',
                              'divisionLeader', 'wildCardLeader', 'eliminationNumber',
                              'wildCardEliminationNumber', 'homeWins', 'homeLosses', 'awayWins',
                              'awayLosses', 'lastTenWins', 'lastTenLosses', 'extraInningWins',
                              'extraInningLosses', 'oneRunWins', 'oneRunLosses', 'dayWins',
                              'dayLosses', 'nightWins', 'nightLosses', 'grassWins', 'grassLosses',
                              'turfWins', 'turfLosses', 'divWins', 'divLosses', 'xWinLossPct',
                              'winStreak', 'loosingStreak', 'xWinLossPctWasMissing', 'hasStandings']


def get_dates_with_info(next_day_player_engagement, seasons):
    dates = pd.DataFrame(data = {'dailyDataDate': next_day_player_engagement['dailyDataDate'].unique()})
    dates['date'] = pd.to_datetime(dates['dailyDataDate'].astype(str))
    dates['year'] = dates['date'].dt.year
    dates['month'] = dates['date'].dt.month
    dates_with_info = pd.merge(
      dates,
      seasons,
      left_on = 'year',
      right_on = 'seasonId'
      )
    dates_with_info['inSeason'] = (
      dates_with_info['date'].between(
        dates_with_info['regularSeasonStartDate'],
        dates_with_info['postSeasonEndDate'],
        inclusive = True
        )
      ).astype("int32")
    dates_with_info['seasonPart'] = np.select(
      [
        dates_with_info['date'] < dates_with_info['preSeasonStartDate'], 
        dates_with_info['date'] < dates_with_info['regularSeasonStartDate'],
        dates_with_info['date'] <= dates_with_info['lastDate1stHalf'],
        dates_with_info['date'] < dates_with_info['firstDate2ndHalf'],
        dates_with_info['date'] <= dates_with_info['regularSeasonEndDate'],
        dates_with_info['date'] < dates_with_info['postSeasonStartDate'],
        dates_with_info['date'] <= dates_with_info['postSeasonEndDate'],
        dates_with_info['date'] > dates_with_info['postSeasonEndDate']
      ], 
      [
        'Offseason',
        'Preseason',
        'Reg Season 1st Half',
        'All-Star Break',
        'Reg Season 2nd Half',
        'Between Reg and Postseason',
        'Postseason',
        'Offseason'
      ], 
      default = np.nan
      )
    dates_with_info["day"] = dates_with_info["date"].dt.day
    dates_with_info["week"] = dates_with_info["date"].dt.isocalendar().week
    dates_with_info["dayOfWeek"] = dates_with_info["date"].dt.dayofweek
    dates_with_info["dayOfYear"] = dates_with_info["date"].dt.dayofyear
    return dates_with_info[['dailyDataDate', 'date', 'year', 'month', 'week', 'day',
                            'dayOfWeek', 'dayOfYear', 'inSeason', 'seasonPart']]


def prepare_players(players):
    players_to_merge = players.loc[:, PLAYERS_FEATURE_NAMES]
    players_to_merge["DOB"] = pd.to_datetime(players["DOB"], format="%Y-%m-%d")
    # TODO: change 2 rows below to a later date
    players_to_merge["playerAge"] = (pd.to_datetime("20211231", format="%Y%m%d") - players_to_merge["DOB"].astype("datetime64")).astype("timedelta64[D]")
    players_to_merge["daysSinceMlbDebute"] = (pd.to_datetime("20211231", format="%Y%m%d") - players_to_merge["mlbDebutDate"].astype("datetime64")).astype("timedelta64[D]")
    players_to_merge["daysSinceMlbDebute"] = players_to_merge["daysSinceMlbDebute"].fillna(players_to_merge["daysSinceMlbDebute"].median())
    players_to_merge["daysSinceMlbDebuteWasMissing"] = players_to_merge["mlbDebutDate"].isna().astype(int)
    players_to_merge.drop(["DOB", "mlbDebutDate"], axis=1, inplace=True)
    return players_to_merge


def prepare_rosters(rosters):
    if rosters.empty:
        return pd.DataFrame()
    rosters_to_merge = rosters.loc[:, ROSTERS_FEATURE_NAMES]
    # aggregate in case of multiple roster rows per player-date
    rosters_to_merge = rosters_to_merge.groupby(['dailyDataDate', 'playerId'], as_index = False).agg(
        {"teamId": "min", "status": "min"})
    rosters_to_merge["hasRoster"] = 1
    rosters_to_merge["teamId"] = rosters_to_merge["teamId"].astype(str)
    rosters_to_merge.rename(columns={"status": "rosterStatus"}, inplace=True)
    return rosters_to_merge


def prepare_teams(teams):
    teams_to_merge = teams.loc[:, TEAMS_FEATURE_NAMES]
    teams_to_merge.rename(columns={"id": "teamId"}, inplace=True)
    teams_to_merge["teamId"] = teams_to_merge["teamId"].astype(str)
    return teams_to_merge


def prepare_events(events, games):
    # Merge games w/ events to get scheduled length of game (helps w/ some calculations)
    events_plus = pd.merge(
      events,
      games[['gamePk', 'scheduledInnings']].drop_duplicates(),
      on = ['gamePk'],
      how = 'left'
    )
    # Get current score from batting & pitching team perspectives
    events_plus['battingTeamScore'] = np.where(events_plus['halfInning'] == 'bottom',
      events_plus['homeScore'], events_plus['awayScore'])

    events_plus['pitchingTeamScore'] = np.where(events_plus['halfInning'] == 'bottom',
      events_plus['awayScore'], events_plus['homeScore'])

    events_plus['pitches100mph'] = np.where(
      (events_plus['type'] == 'pitch') & (events_plus['startSpeed'] >= 100),
      1, 0)

    events_plus['HRDist450ft'] = np.where(
      (events_plus['event'] == 'Home Run') & (events_plus['totalDistance'] >= 450),
      1, 0)

    # Use game context/score logic to add fields for notable in-game events
    events_plus['gameTyingRBI'] = np.where(
      (events_plus['isPaOver'] == 1) & (events_plus['rbi'] > 0) &
      # Start w/ batting team behind in score...
      (events_plus['battingTeamScore'] < events_plus['pitchingTeamScore']) &
      # ...and look at cases where adding RBI ties score
      ((events_plus['battingTeamScore'] + events_plus['rbi']) ==
        events_plus['pitchingTeamScore']
        ),
      1, 0)

    events_plus['goAheadRBI'] = np.where(
      (events_plus['isPaOver'] == 1) & (events_plus['rbi'] > 0) &
      # Start w/ batting team not ahead in score (can be tied)...
      (events_plus['battingTeamScore'] <= events_plus['pitchingTeamScore']) &
      # ... and look at cases where adding RBI puts batting team ahead
      ((events_plus['battingTeamScore'] + events_plus['rbi']) >
        events_plus['pitchingTeamScore']
        ),
      1, 0)

    # Add field to count walk-off (game-winning, game-ending) RBI
    events_plus['walkoffRBI'] = np.where(
      (events_plus['inning'] >= events_plus['scheduledInnings']) &
      (events_plus['halfInning'] == 'bottom') &
      (events_plus['goAheadRBI'] == 1),
      1, 0)

    added_events_fields = ['pitches100mph', 'HRDist450ft', 'gameTyingRBI', 'goAheadRBI', 'walkoffRBI']

    # Aggregate player event-based stats to player-date level
    pitcher_date_events_agg = (events_plus.
      groupby(['dailyDataDate', 'pitcherId'], as_index = False).
      agg(
        pitches100mph = ('pitches100mph', 'sum'),
        walkoffRBIAllowed = ('walkoffRBI', 'sum')
        )
      )

    hitter_date_events_agg = (events_plus.
      groupby(['dailyDataDate', 'hitterId'], as_index = False)
      [[field for field in added_events_fields if field != 'pitches100mph']].
      sum()
      )

    events_to_merge = (pd.merge(
      pitcher_date_events_agg.rename(columns = {'pitcherId': 'playerId'}),
      hitter_date_events_agg.rename(columns = {'hitterId': 'playerId'}),
      on = ['dailyDataDate', 'playerId'],
      how = 'outer'
      ).
      # NAs on events fields can be turned to 0 (no such stats in those categories)
      fillna({field: 0 for field in added_events_fields + ['walkoffRBIAllowed']})
      )
    events_to_merge["hasEvents"] = 1
    return events_to_merge


def prepare_standings(standings):
    standings_to_merge = standings.loc[:, STANDINGS_FEATURE_NAMES]
    standings_to_merge["streakCode"] = standings_to_merge["streakCode"].fillna("UNK")
    standings_to_merge["winStreak"] = standings_to_merge["streakCode"].apply(lambda x: int(x[1:]) if x[0] == "W" else 0)
    standings_to_merge["loosingStreak"] = standings_to_merge["streakCode"].apply(lambda x: int(x[1:]) if x[0] == "L" else 0)
    standings_to_merge.drop("streakCode", axis=1, inplace=True)
    
    standings_to_merge["divisionLeader"] = standings_to_merge["divisionLeader"].astype(int)
    standings_to_merge["divisionChamp"] = standings_to_merge["divisionChamp"].astype(int)
    standings_to_merge["wildCardLeader"].replace({None: "None", np.nan: "UNK"}, inplace=True)
    
    x_win_loss_pct_missing = standings_to_merge["xWinLossPct"].isna()
    standings_to_merge["xWinLossPct"].fillna(0.5, inplace=True)
    standings_to_merge["xWinLossPctWasMissing"] = x_win_loss_pct_missing.astype(int)
    
    standings_to_merge["divisionRank"].fillna("UNK", inplace=True)
    standings_to_merge["wildCardRank"].fillna("UNK", inplace=True)
    standings_to_merge["leagueRank"].fillna("UNK", inplace=True)
    standings_to_merge["divisionRank"] = standings_to_merge["divisionRank"].astype(str)
    standings_to_merge["leagueRank"] = standings_to_merge["leagueRank"].astype(str)
    standings_to_merge["wildCardRank"] = standings_to_merge["wildCardRank"].astype(str)
    standings_to_merge["divisionRank"] = standings_to_merge["divisionRank"].apply(lambda x: x.split(".")[0])
    standings_to_merge["leagueRank"] = standings_to_merge["leagueRank"].apply(lambda x: x.split(".")[0])
    standings_to_merge["wildCardRank"] = standings_to_merge["wildCardRank"].apply(lambda x: x.split(".")[0])
    
    standings_to_merge["eliminationNumber"] = standings_to_merge["eliminationNumber"].fillna("UNK")
    standings_to_merge["wildCardEliminationNumber"] = standings_to_merge["wildCardEliminationNumber"].fillna("UNK")
    
    standings_to_merge.rename(columns={"runsScored": "runsScoredStandings"}, inplace=True)
    
    standings_to_merge["teamId"] = standings_to_merge["teamId"].astype(str)
    standings_to_merge["hasStandings"] = 1
    
    standings_to_merge = standings_to_merge.groupby(["dailyDataDate", "teamId"], as_index=False).agg("min")
    return standings_to_merge


def prepare_awards(awards):
    awards_to_merge = awards.loc[:, AWARDS_FEATURE_NAMES]
    awards_to_merge = awards_to_merge.groupby(["dailyDataDate", "playerId"], as_index=False).agg(
        numAwards = ("awardId", "nunique"),
        awardId = ("awardId", "min"),
        awardName = ("awardName", "min")
    )
    awards_to_merge["awardCategory"] = awards_to_merge["awardName"].apply(lambda x: AWARD_CATEGORIES[x] if x in AWARD_CATEGORIES else "other")
    awards_to_merge.drop(["awardName", "awardId"], axis=1, inplace=True)
    awards_to_merge["hasAwards"] = 1
    return awards_to_merge


def prepare_transactions(transactions):
    transactions = transactions.loc[:, TRANSACTIONS_FEATURE_NAMES]
    transactions = transactions.dropna(subset=["playerId"])
    transactions["playerId"] = transactions["playerId"].astype(int)
    transactions["date"] = pd.to_datetime(transactions["date"], format="%Y-%m-%d")
    transactions["effectiveDate"] = pd.to_datetime(transactions["effectiveDate"], format="%Y-%m-%d")
    transactions["resolutionDate"] = pd.to_datetime(transactions["resolutionDate"], format="%Y-%m-%d")
    transactions["effectiveDateDelta"] = (transactions["effectiveDate"] - transactions["date"]).dt.days
    transactions["resolutionDateDelta"] = (transactions["resolutionDate"] - transactions["date"]).dt.days
    transactions.drop(["date", "effectiveDate", "resolutionDate"], axis=1, inplace=True)
    
    transaction_types_of_interest = transactions["typeDesc"].unique()
    player_date_transactions_wide = (transactions.
      assign(
        # Create field w/ initial lower case & w/o spaces for later field names
        typeDescNoSpace = [(typeDesc[0].lower() + typeDesc[1:]) for typeDesc in
          transactions['typeDesc'].str.replace(' ', '')],
        # Add count ahead of pivot
        count = 1
        )
      [
      # Filter to transactions of desired types and rows for actual players
        np.isin(transactions['typeDesc'], transaction_types_of_interest) &
        pd.notna(transactions['playerId'])
      ][['dailyDataDate', 'playerId', 'typeDescNoSpace', 'count']].
      # Filter to unique transaction types across player-date
      drop_duplicates().
      # Pivot data to 1 row per player-date and 1 column per transaction type
      pivot_table(
        index = ['dailyDataDate', 'playerId'],
        columns = 'typeDescNoSpace',
        values = 'count',
        # NA can be turned to 0 since it means player didn't have that transaction that day
        fill_value = 0
        ).
      reset_index()
      )
    transactions_to_merge = transactions.groupby(["dailyDataDate", "playerId"], as_index=False).agg(
        fromTeamId = ("fromTeamId", "min"),
        toTeamId = ("toTeamId", "min"),
        numTransactionInfos = ("transactionId", "nunique"),
        effectiveDateDelta = ("effectiveDateDelta", "min"),
        resolutionDateDelta = ("resolutionDateDelta", "min")
    )
    transactions_to_merge = transactions_to_merge.merge(player_date_transactions_wide, on=["dailyDataDate", "playerId"], how="left")
    transactions_to_merge["hasTransactions"] = 1
    transactions_to_merge.fillna({
        "fromTeamId": "UNK",
        "toTeamId": "UNK",
        "resolutionDateDelta": 0,
    }, inplace=True)
    transactions_to_merge["fromTeamId"] = transactions_to_merge["fromTeamId"].astype(str).apply(lambda x: x.split(".")[0])
    transactions_to_merge["toTeamId"] = transactions_to_merge["toTeamId"].astype(str).apply(lambda x: x.split(".")[0])
    missing_cols = list(set(TRANSACTION_DESC_COLS) - set(transactions_to_merge.columns))
    transactions_to_merge[missing_cols] = np.nan
    return transactions_to_merge


def prepare_player_box_scores(player_box_scores):
    player_game_stats = player_box_scores.loc[:, PLAYER_BOX_SCORES_FEATURE_NAMES]
    # Adds in field for innings pitched as fraction (better for aggregation)
    player_game_stats['inningsPitchedAsFrac'] = np.where(pd.isna(player_game_stats['inningsPitched']),
                                                         np.nan, np.floor(player_game_stats['inningsPitched']) +
                                                         (player_game_stats['inningsPitched'] -
                                                          np.floor(player_game_stats['inningsPitched'])) * 10/3)
    # Add in Tom Tango pitching game score (https://www.mlb.com/glossary/advanced-stats/game-score)
    player_game_stats['pitchingGameScore'] = np.where(
        # pitching game score doesn't apply if player didn't pitch, set to NA
        pd.isna(player_game_stats['pitchesThrown']) | (player_game_stats['pitchesThrown'] == 0), np.nan,
        (40
         + 2 * player_game_stats['outsPitching']
         + 1 * player_game_stats['strikeOutsPitching']
         - 2 * player_game_stats['baseOnBallsPitching']
         - 2 * player_game_stats['hitsPitching']
         - 3 * player_game_stats['runsPitching']
         - 6 * player_game_stats['homeRunsPitching']
        )
    )
    # Add in criteria for no-hitter by pitcher (individual, not multiple pitchers)
    player_game_stats['noHitter'] = np.where(
        (player_game_stats['completeGamesPitching'] == 1) &
        (player_game_stats['inningsPitched'] >= 9) &
        (player_game_stats['hitsPitching'] == 0), 1, 0)
    
    player_game_stats["battingOrderSpot"] = player_game_stats.battingOrder.astype("str").apply(lambda x: x[0]).replace("n", np.nan).astype("float64")
    player_game_stats["battingOrderSequence"] = player_game_stats.battingOrder.astype("str").apply(lambda x: x[1:3]).replace("an", np.nan).astype("float64")
    player_game_stats.drop("battingOrder", axis=1, inplace=True)
    
    player_box_scores_to_merge = pd.merge(
        player_game_stats.groupby(["dailyDataDate", "playerId"], as_index=False).agg(
            {'positionCode': "min", 'positionType': "min"}),
        player_game_stats.groupby(["dailyDataDate", "playerId"], as_index=False)[[col for col in player_game_stats.columns if col not in ["dailyDataDate", "playerId", "positionCode", "positionType"]]].sum(),
        on=["dailyDataDate", "playerId"], how="inner")
    player_box_scores_to_merge["hasPlayerBoxScores"] = 1
    
    player_box_scores_to_merge["positionCode"] = player_box_scores_to_merge["positionCode"].astype(str)
    return player_box_scores_to_merge


def prepare_team_box_scores(team_box_scores):
    team_box_scores_to_merge = team_box_scores.loc[:, TEAM_BOX_SCORES_FEATURE_NAMES]
    team_box_scores_to_merge.columns = [(col_value + 'Team') if (col_value not in ['dailyDataDate', 'teamId']) else col_value
                                        for col_value in team_box_scores_to_merge.columns.values]
    team_box_scores_to_merge = team_box_scores_to_merge.groupby(["dailyDataDate", "teamId"], as_index=False).agg("mean")
    team_box_scores_to_merge["teamId"] = team_box_scores_to_merge["teamId"].astype(str)
    team_box_scores_to_merge["hasTeamBoxScores"] = 1
    return team_box_scores_to_merge


def prepare_games(games):
    home_games = games.loc[:, GAMES_FEATURE_NAMES]
    away_games = games.loc[:, GAMES_FEATURE_NAMES]
    home_games["isHome"] = 1
    away_games["isHome"] = 0
    home_games.columns = [col_value.replace("home", "team").replace("away", "opp")
                          for col_value in home_games.columns.values]
    away_games.columns = [col_value.replace("home", "opp").replace("away", "team")
                          for col_value in away_games.columns.values]
    games_to_merge = pd.concat([home_games, away_games], ignore_index=True)
    games_to_merge["gameHour"] = games_to_merge["gameTimeUTC"].astype("datetime64").dt.hour.astype(str)
    games_to_merge["isNight"] = (games_to_merge["dayNight"] == "night").astype(float)
    games_to_merge["isTie"] = (games_to_merge["isTie"] == "night").astype(float)
    games_to_merge["hasGame"] = 1
    games_to_merge.drop(["gameTimeUTC", "dayNight"], axis=1, inplace=True)
    games_to_merge = games_to_merge.groupby(["dailyDataDate", "teamId"], as_index=False).agg(
        {
            "gamePk": "min",
            "gameType": "min",
            "detailedGameState": "min",
            "isTie": "max",
            "gameNumber": "max",
            "doubleHeader": "min",
            "scheduledInnings": "mean",
            "gamesInSeries": "mean",
            "seriesDescription": "min",
            "teamWins": "mean",
            "teamLosses": "mean",
            "teamWinPct": "mean",
            "teamWinner": "max",
            "teamScore": "mean",
            "oppId": "min",
            "oppWins": "mean",
            "oppLosses": "mean",
            "oppWinPct": "mean",
            "oppWinner": "max",
            "oppScore": "mean",
            "isHome": "max",
            "gameHour": "min",
            "isNight": "max",
            "hasGame": "max",
        },
    )
    games_to_merge["teamId"] = games_to_merge["teamId"].astype(str)
    games_to_merge["oppId"] = games_to_merge["oppId"].astype(str)
    games_to_merge["teamWinner"] = games_to_merge["teamWinner"].astype(float)
    games_to_merge["oppWinner"] = games_to_merge["oppWinner"].astype(float)
    return games_to_merge


def get_player_target_stats(next_day_player_engagement):
    # TODO compute target stats per year, need to update get_lagged_targets_test as it relies on it
    player_target_stats = next_day_player_engagement.loc[:, ["playerId", "target1", "target2", "target3", "target4"]]
    player_target_stats = player_target_stats.groupby("playerId").describe()
    player_target_stats.columns = player_target_stats.columns.map("_".join)
    player_target_stats.drop(["target1_count", "target2_count", "target3_count", "target4_count"], axis=1, inplace=True)
    player_target_stats.reset_index(inplace=True)
    player_target_stats.columns = [col.replace("%", "") for col in player_target_stats.columns]
    return player_target_stats


def prepare_mlb_social_media(mlb_social_media):
    def abbrev_to_float(x):
        if x[-1] in ["k", "K"]:
            return float(x[:-1]) * 1000
        elif x[-1] in ["m", "M"]:
            return float(x[:-1]) * 1000000
        else:
            return float(x)
    mlb_social_media = mlb_social_media.loc[:, MLB_SOCIAL_MEDIA_FEATURE_NAMES]
    # missing values
    houston_astros = {"id": 117, "tik_followers": "24.5K", "tik_likes": "0"}
    washington_nationals = {"id": 120, "tik_followers": "13.8K", "tik_likes": "16.6K"}
    miami_marlins = {"id": 146, "tik_followers": "0", "tik_likes": "0"}
    for team in [houston_astros, washington_nationals, miami_marlins]:
        for col_name in ["tik_followers", "tik_likes"]:
            mlb_social_media.loc[mlb_social_media[mlb_social_media["id"] == team["id"]].index[0], col_name] = team[col_name]
    str_cols = mlb_social_media.columns[mlb_social_media.dtypes == "object"]
    for col_name in str_cols:
        mlb_social_media[col_name] = mlb_social_media[col_name].apply(lambda x: abbrev_to_float(x))
    mlb_social_media["hasSocialMediaData"] = 1
    mlb_social_media["teamId"] = mlb_social_media["id"].astype(str)
    mlb_social_media.drop("id", axis=1, inplace=True)
    return mlb_social_media


def prepare_player_salaries(player_salaries, players):
    unique_players = players["playerId"].unique()
    n_players = players["playerId"].nunique()
    years = player_salaries["year"].unique()
    player_salaries.rename(columns={"playerID": "playerId"}, inplace=True)
    player_salaries = player_salaries[player_salaries["playerId"].isin(unique_players)]
    player_salaries = player_salaries[["playerId", "year", "salary"]].groupby(["playerId", "year"], as_index=False).agg(max)
    player_salaries_to_merge = pd.DataFrame({"year": list(chain.from_iterable([year]*n_players for year in years)),
                                             "playerId": list(chain.from_iterable(players["playerId"].unique() for _ in range(len(years))))})
    player_salaries_to_merge = player_salaries_to_merge.merge(player_salaries[["year", "playerId", "salary"]],
                                                              on=["playerId", "year"], how="left")
    player_salaries_to_merge["salaryWasMissing"] = player_salaries_to_merge["salary"].isna().astype(int)
    player_salaries_to_merge["salary"].fillna(player_salaries_to_merge["salary"].min(), inplace=True)
    player_salaries_to_merge["hasSalary"] = 1
    return player_salaries_to_merge


def get_lagged_targets_train(next_day_player_engagement, lag):    
    shifted_engagement = next_day_player_engagement.copy()
    shifted_engagement["dailyDataDate"] = pd.to_datetime(shifted_engagement["dailyDataDate"], format="%Y%m%d") + pd.Timedelta(days=lag)
    shifted_engagement["dailyDataDate"] = shifted_engagement["dailyDataDate"].dt.strftime("%Y%m%d").astype(int)
    
    lagged_targets = next_day_player_engagement.loc[:, ["dailyDataDate", "playerId"]]
    lagged_targets = lagged_targets.merge(shifted_engagement, on=["dailyDataDate", "playerId"], how="left")
    lagged_targets.drop("engagementMetricsDate", axis=1, inplace=True)
    
    target_medians = next_day_player_engagement.groupby("playerId", as_index=False).median()
    defaults = lagged_targets.loc[:, ["dailyDataDate", "playerId"]]
    defaults = defaults.merge(target_medians, on=["playerId"], how="left")  # use median as a default
    mask = lagged_targets.isna()
    lagged_targets = lagged_targets.where(~mask, other=defaults.where(mask))
        
    lagged_targets.rename(columns={
        "target1": f"target1Lag{lag}",
        "target2": f"target2Lag{lag}",
        "target3": f"target3Lag{lag}",
        "target4": f"target4Lag{lag}",
    }, inplace=True)
    return lagged_targets


def get_lagged_targets_test(sample_submission, historic_targets, player_target_stats, lag):
    lagged_targets = pd.DataFrame(sample_submission.loc[:, "date_playerId"])
    lagged_targets["playerId"] = lagged_targets["date_playerId"].apply(lambda x: x.split("_")[1]).astype(int)
    lagged_targets.reset_index(inplace=True)
    lagged_targets.rename(columns={"date": "dailyDataDate"}, inplace=True)
    lagged_targets.drop(["date_playerId"], axis=1, inplace=True)

    earliest_date = int((pd.to_datetime(lagged_targets["dailyDataDate"].min(), format="%Y%m%d") - pd.Timedelta(days=lag)).strftime("%Y%m%d"))

    shifted_historic_targets = historic_targets[historic_targets["dailyDataDate"] >= earliest_date].copy()
    shifted_historic_targets["dailyDataDate"] = pd.to_datetime(shifted_historic_targets["dailyDataDate"], format="%Y%m%d") + pd.Timedelta(days=lag)
    shifted_historic_targets["dailyDataDate"] = shifted_historic_targets["dailyDataDate"].dt.strftime("%Y%m%d").astype(int)

    lagged_targets = pd.merge(lagged_targets, shifted_historic_targets, on=["dailyDataDate", "playerId"], how="left")

    defaults = lagged_targets.loc[:, ["dailyDataDate", "playerId"]]
    target_medians = player_target_stats[["playerId", "target1_50", "target2_50", "target3_50", "target4_50"]].rename(columns={
        "target1_50": "target1", "target2_50": "target2", "target3_50": "target3", "target4_50": "target4"
    })
    # TODO: update after changing to yearly stats
    defaults = defaults.merge(target_medians, on=["playerId"], how="left")  # use median as a default

    mask = lagged_targets.isna()
    lagged_targets = lagged_targets.where(~mask, other=defaults.where(mask))

    lagged_targets.rename(columns={
        "target1": f"target1Lag{lag}",
        "target2": f"target2Lag{lag}",
        "target3": f"target3Lag{lag}",
        "target4": f"target4Lag{lag}",
    }, inplace=True)
    return lagged_targets


def get_dataset_from_df(df, batch_size, is_test=False):
    #df = df.copy()
    if is_test:
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    else:
        labels = df[TARGET_FEATURE_NAMES]
        df.drop(TARGET_FEATURE_NAMES, axis=1, inplace=True)
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    return dataset.batch(batch_size)
