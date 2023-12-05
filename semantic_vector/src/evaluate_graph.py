import os
import csv
import pandas as pd
import networkx as nx
import gensim
from string import ascii_letters as al

def expectlatin(mystring):
  # Function checks for latin words. Temporary sollution.
  result = ''
  for i in mystring:
    if i not in al:
      result += i
  if result != '':
    return result
  else:
    return 'абв'
  
current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
models_path = os.path.join(current_dir, 'models')
data_path = os.path.join(current_dir, 'training_data')

custom_model_path = os.path.join(models_path, 'default_trained.model')
print(custom_model_path)
# open model
model_trained = gensim.models.Word2Vec.load(custom_model_path)

# open graph
graph_path = os.path.join(data_path, 'kg.graphml')

G = nx.read_graphml(graph_path)

# open as dataframe
df = nx.to_pandas_edgelist(G)

# create and write csv
output_file_path = os.path.join(current_dir, 'output_data')
csv_path = os.path.join(output_file_path, 'semantic_value.csv')

with open(csv_path, mode='w', encoding='utf-8') as file:
  
    # intiate counters
    sucess = 0
    fail = 0

    # establish columns names
    names = ['source', 'target', 'semantic closeness']

    #create a writer
    file_writer = csv.DictWriter(file, delimiter = ";", lineterminator="\r", fieldnames=names)

    for i in range(len(df)):

        # make a list of pair words
        words = [df['source'][i].lower(), df['target'][i].lower()]
        # cleaning words in created list
        words = [expectlatin(''.join(c for c in word if c.isalpha())) for word in words]

        for word in words:
            # checking for word in model
            if word in model_trained.wv.key_to_index:
                check_status = True
            else:
                fail += 1
                check_status = False
                break

    #checking results and writing it in csv
        if check_status == True:
            sucess += 1
            print(words)
            custom_result = model_trained.wv.similarity(word[0], word[1])
            print(custom_result)
            file_writer.writerow({'source': words[0], 'target': words[1], 'semantic closeness': custom_result})
      

    print(f'Успешных оценок: {sucess}')
    print(f'Провальных оценок: {fail}')
