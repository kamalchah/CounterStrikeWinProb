import pandas as pd
from glob import glob
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

ds_collection_path = os.path.join('..','..','CSDScollection') 
ds_type = 'csds'
ds_collection_path = os.path.abspath(ds_collection_path)

def extract_and_add_to_csds(base_folder):
    result = {}

    for root, dirs, files in os.walk(base_folder):
        for folder in dirs:
            folder_path = os.path.join(root, folder) 
            text_files = []

            vectorFileChecker = 0
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                try:
                    df=pd.read_parquet(file_path)
                    if file in ['player_vector', 'player_status', 'round_end', 'header']:
                      vectorFileChecker += 1
                       
                    if(file=='header' and not(pd.read_parquet(file_path)['map_name'].to_list()[0]=='de_mirage')):
                       continue


                    text_files.append({'filename': file, 'dataframe': df})
                except Exception as e:
                    continue
                
            if vectorFileChecker >= 4 and text_files:
              result[folder] = text_files

    return result

# your location of the stored demos
dir_name = "C:\\Users\\kAMAL\\Desktop\\pureskill\\\Demos"
dir_name = "C:\\Users\\kAMAL\\Desktop\\pureskill\\\demoOne"
cs = extract_and_add_to_csds(dir_name)

T_TEAM_CODE = 2
CT_TEAM_CODE = 3
random_tick=0

# Get Alive count for team
def get_alive_count(alive_data, team_code, tick):
  try:
    return alive_data.loc[random_tick,team_code]
  except KeyboardInterrupt:
    raise
  except KeyError:
    return 0

# Get equipment value for team
def get_eq_val(equipment_data, team_code, tick):
  try:
    return equipment_data.loc[random_tick,team_code]
  except KeyboardInterrupt:
    raise
  except KeyError:
    return 0
  
# Create word embedding string using  x and y pos, dividing by 256
def create_string(tPos,ctPos):
    if len(tPos)== 0 and len(ctPos) == 0:
        return "0,0a0,0a0,0a0,0a0,0a_0,0a0,0a0,0a0,0a0,0a"

    string_xCT = ""
    fifthIterationChecker = 0

    if(len(ctPos)>0):
       for index, row in ctPos.iterrows():
        string_xCT += f"{round(row['x_pos'] / 256)},{round(row['y_pos'] / 256)}"
        fifthIterationChecker+=1
        if(fifthIterationChecker!=5): string_xCT +="a"
        else: string_xCT += "_" 
       if(fifthIterationChecker<5):
          for i in range(fifthIterationChecker,5):
             if(i!=4):string_xCT +="0,0a"
             else:string_xCT +="0,0a_"
       fifthIterationChecker=0
    else:string_xCT="0,0a0,0a0,0a0,0a0,0a_"
       
    if(len(tPos)>0):
       for index, row in tPos.iterrows():
        string_xCT += f"{round(row['x_pos'] / 256)},{round(row['y_pos'] / 256)}"
        fifthIterationChecker+=1
        string_xCT +="a"
       if(fifthIterationChecker<5):
          for i in range(fifthIterationChecker,5):
             string_xCT +="0,0a"
    else:string_xCT+="0,0a0,0a0,0a0,0a0,0a"

    return string_xCT

# Get round winner
def get_round_winner(round_end, rnd):
    try:
        return 1 if round_end[round_end['round']==rnd]['winner_team_code'].iloc[0] == CT_TEAM_CODE else 0
    except KeyboardInterrupt:
        raise
    except IndexError:
        return None

# Get a specific file dataframe from a demo, player_vector, player_status etc...
def getDataFrame(dfList,dfName):
   selected_dataframe = None  # Initialize with None in case the filename is not found
   for entry in dfList:
      if entry.get('filename') == dfName:
          selected_dataframe = entry.get('dataframe')
          return selected_dataframe 

    
def getSentencesForMatch(match):
    player_vector = getDataFrame(match,'player_vector')
    mapFile = getDataFrame(match,'header')
    tick_rate = round(mapFile['tick_rate'].to_list()[0])
    # print(tick_rate) # map tick rate 64/128

    seconds = 3
    skip_interval = tick_rate*seconds

    rounds = list(set(player_vector['round']))
    first = player_vector[(player_vector['x_vel'] > 0) | (player_vector['y_vel'] > 0) | (player_vector['z_vel'] > 0)].groupby('round').first().reset_index()
    grandMasterSentence = []

    # we get a sentence from each round, word embedding is taken every 3 seconds of a round after it starts
    for rnd in rounds:
        firstMovementTick = first[first['round']==rnd] # we start indexing here

        currentRoundTicks = player_vector[(player_vector['round'] == rnd)] 
        sentence = []
        

        for _tick in range(currentRoundTicks['tick'].min(), currentRoundTicks['tick'].max() - 1, skip_interval):
            tick=_tick
            max_tick_check=10
            while max_tick_check > 0:
                if tick in currentRoundTicks['tick'].to_list():
                    break
                tick += 1
                max_tick_check -= 1
            if max_tick_check == 0:
              continue

            currentTickRows = currentRoundTicks[currentRoundTicks['tick'] == tick].copy()
            
            if(currentTickRows['tick'].values.size==0 or firstMovementTick['tick'].values.size==0):
               continue
            if any(currentTickRows['tick'].values < firstMovementTick['tick'].values):
               continue
            
            TSideForTick = currentTickRows[currentTickRows['team_code']==2]
            CTSideForTick = currentTickRows[currentTickRows['team_code']==3]

            tPositions = CTSideForTick[['x_pos', 'y_pos']].copy()
            ctPositions = TSideForTick[['x_pos', 'y_pos']].copy()
            sentence.append(create_string(tPositions,ctPositions))
        grandMasterSentence.append(sentence)

    return grandMasterSentence

allMatchesWordEmbed = []
matches = list(cs.values())
newMatches = []

# Making sure we have valid demo files with all required parquet's
for i, match in enumerate(matches):
    player_vector_df = getDataFrame(match, 'player_vector')
    player_status_df = getDataFrame(match, 'player_status')
    header_df = getDataFrame(match, 'header')
    round_end_df = getDataFrame(match, 'round_end')
    unique_rounds_list = player_vector_df['round'].unique().tolist()

    if not all(round_num in unique_rounds_list for round_num in [2, 4, 6, 8]):
      print("skipped")
      continue
    if player_vector_df is not None and player_status_df is not None and header_df is not None and round_end_df is not None:
        newMatches.append(match)

# Extracting sentences from valid demos to train word2vec
for i, match in enumerate(newMatches):
  player_vector = getDataFrame(match,'player_vector')
  selected_data = []

  # Find the first instance of each round with a moving player
  first_instance_per_round = player_vector[(player_vector['x_vel'] > 0) | (player_vector['y_vel'] > 0) | (player_vector['z_vel'] > 0)].groupby('round').first() 
  selected_rows = [] 
  
  fullList = getSentencesForMatch(match)
  allMatchesWordEmbed.extend(fullList)

# Training word2vec
model = Word2Vec(sentences=allMatchesWordEmbed, vector_size=8, window=5, min_count=1)
model.save("word2vec.model")


datasetWithWordEmbed=[]
datasetNormal=[]

# Extracting datasets with and without word embeddings
for i, match in enumerate(newMatches):
  player_vector = getDataFrame(match,'player_vector')
  player_status = getDataFrame(match,'player_status')
  round_end= getDataFrame(match,'round_end')

  rounds = list(set(player_vector['round']))
  alive_data = player_vector.groupby(['tick','team_code']).size()

  player_status=pd.merge(player_status,player_vector[['tick','player_id_fixed','team_code']],on=['tick','player_id_fixed'])
  equipment_data = player_status.groupby(['tick','team_code'])['equipment_value_calc'].sum()

  for rnd in rounds:
    random_tick = player_vector[player_vector['round']==rnd]['tick'].sample(n=1).to_list()[0]
    
    # Information for current tick
    currentTick = player_vector[(player_vector['round'] == rnd) & (player_vector['tick'] == random_tick)]
    TSideForTick = currentTick[currentTick['team_code']==2]
    CTSideForTick = currentTick[currentTick['team_code']==3]
    tPositions = CTSideForTick[['x_pos', 'y_pos']].copy()
    ctPositions = TSideForTick[['x_pos', 'y_pos']].copy()

    # Creating word vector for current tick and player positions
    word = create_string(tPositions,ctPositions)

    try:  
      word_vector = model.wv[word]
    except:
      continue
    
    print("Vector for 'example_word':", word_vector)

    # Dataset with word embeddings
    datasetWithWordEmbed.append(
    {
        "num_t_alive": get_alive_count(alive_data, T_TEAM_CODE, random_tick),
        "num_ct_alive": get_alive_count(alive_data, CT_TEAM_CODE, random_tick),
        "t_equip_value": get_eq_val(equipment_data, T_TEAM_CODE, random_tick),
        "position_embeddings_v1":word_vector[0],
        "position_embeddings_v2":word_vector[1],
        "position_embeddings_v3":word_vector[2],
        "position_embeddings_v4":word_vector[3],
        "position_embeddings_v5":word_vector[4],
        "position_embeddings_v6":word_vector[5],
        "position_embeddings_v7":word_vector[6],
        "position_embeddings_v8":word_vector[7],
        "ct_equip_value": get_eq_val(equipment_data, CT_TEAM_CODE, random_tick),
        "did_ct_win": 1 if round_end[round_end['round']==rnd]['winner_team_code'].iloc[0] == CT_TEAM_CODE else 0
    })

    # Dataset without word embeddings
    datasetNormal.append(
    {
        "num_t_alive": get_alive_count(alive_data, T_TEAM_CODE, random_tick),
        "num_ct_alive": get_alive_count(alive_data, CT_TEAM_CODE, random_tick),
        "t_equip_value": get_eq_val(equipment_data, T_TEAM_CODE, random_tick),
        "ct_equip_value": get_eq_val(equipment_data, CT_TEAM_CODE, random_tick),
        "did_ct_win": 1 if round_end[round_end['round']==rnd]['winner_team_code'].iloc[0] == CT_TEAM_CODE else 0
    })


dfWordEmbed = pd.DataFrame(datasetWithWordEmbed)
dfNormal = pd.DataFrame(datasetNormal)
# df['did_ct_win'].value_counts()
# df['num_t_alive'].value_counts()
# df['num_ct_alive'].value_counts()
# df['ct_equip_value'].hist()


X_train, X_test, y_train, y_test = train_test_split(dfWordEmbed.drop('did_ct_win',axis=1), dfWordEmbed['did_ct_win'], test_size=.2) # 
print("------------------------------------------")
print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
print("------------------------------------------")

bst = XGBClassifier(max_depth=8,eval_metric='logloss',objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
cm = confusion_matrix(y_test, preds)

print(" Findings with word/position embeddings: ")
print(" Findings with word/position embeddings: ")

from sklearn.metrics import confusion_matrix, f1_score

# Calculate F1 score
f1 = f1_score(y_test, preds)

# Print F1 score
print('F1 Score =', f1)

print({
    'true negatives': round(100*cm[0, 0]/len(preds)),
    'false positives': round(100*cm[0, 1]/len(preds)),
    'false negatives': round(100*cm[1, 0]/len(preds)),
    'true positives': round(100*cm[1, 1]/len(preds))
})

print('accuracy = ',(cm[1, 1]+cm[0, 0])/len(preds))

for name,importance in zip(bst.feature_names_in_,bst.feature_importances_):
  print(name,importance)

print("\n\n")
print(" Findings WITHOUT word/position embeddings: ")
print(" Findings WITHOUT word/position embeddings: ")

X_train2 = X_train.drop([col for col in X_train.columns if col.startswith("position")],axis=1)
X_test2 = X_test.drop([col for col in X_test.columns  if col.startswith("position")],axis=1)
y_train2 = y_train 
y_test2 = y_test 

print("------------------------------------------")
print("X_train2: ",X_train2.shape)
print("X_test2: ",X_test2.shape)
print("y_train2: ",y_train2.shape)
print("y_test2: ",y_test2.shape)
print("------------------------------------------")

bst2 = XGBClassifier(max_depth=8,eval_metric='logloss',objective='binary:logistic')
bst2.fit(X_train2, y_train2)
preds2 = bst2.predict(X_test2)
cm2 = confusion_matrix(y_test2, preds2)

print({
    'true negatives': round(100*cm2[0, 0]/len(preds2)),
    'false positives': round(100*cm2[0, 1]/len(preds2)),
    'false negatives': round(100*cm2[1, 0]/len(preds2)),
    'true positives': round(100*cm2[1, 1]/len(preds2))
})

print('accuracy = ',(cm2[1, 1]+cm2[0, 0])/len(preds2))

for name,importance in zip(bst2.feature_names_in_,bst2.feature_importances_):
  print(name,importance)

print("Number of mirage games analzed: ",len(newMatches))


# Calculate F1 score
f1 = f1_score(y_test2, preds2)

# Print F1 score
print('F1 Score =', f1)