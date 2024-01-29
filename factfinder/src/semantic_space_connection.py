import pandas as pd

import geocoder


df = pd.read_csv('data\\raw\\posts_spb_today.csv', delimiter=';').dropna(subset='text')
print(df)
g = geocoder.Geocoder()
names_df = g.run(df, text_column='text')

print(g)

'''
g = geocoder.Geocoder()
clean_sents = []
p = semantic_model.Preprocessor(filepath='data\\raw\\posts_spb_today.csv', column='text')
data = pd.read_csv('data\\raw\\posts_spb_today.csv', delimiter=';').dropna(subset='text')

print("Parsing sentences from training set...")

# Text clearing
for review in data['text']:
    clean_sents += p.review_to_sentence(review)

df = pd.DataFrame(columns=['Street','Score'])
for i in range(len(clean_sents)):
    for j in range(len(clean_sents[i])):
        tmp = g.extract_ner_street(clean_sents[i][j])
        
        if tmp[0] != None or tmp[1] != None:
            df.loc[len(df.index)]=([tmp[0], tmp[1]])


model = semantic_model.Semantic_model(file_path='data\\raw\\posts_spb_today.csv', column='text').make_model()

semantic_df = pd.DataFrame(columns=['subject','object','semantic_closeness'])
for i in df['Street']:
    for j in df['Street']:
        if i != j:
            semantic_df.loc[len(df.index)]=([i, j, model.wv.similarity(i, j)])

semantic_df = semantic_df[semantic_df['semantic_closeness'] > 0.5]

G=nx.from_pandas_edgelist(semantic_df,"subject","object", edge_attr=True, create_using=nx.MultiDiGraph())

# Visualisation into an image
plt.figure(figsize=(30,30))
pos = nx.spring_layout(G)

options = {'node_color': 'yellow',     # Цвет узлов
                                                                    'node_size': 1000,          # Размер узлов
                                                                    'width': 1,                 # Ширина линий связи
                                                                    'arrowstyle': '-|>',        # Стиль стрелки для напрвленного графа
                                                                    'arrowsize': 18,            # Размер стрелки
                                                                    'edge_color':'blue',        # Цвет связи
                                                                    'font_size':20              # Размер шрифта
                                                                    }

# Visualising the graph into an image and saving it
nx.draw(G, with_labels=True, pos = pos, **options)
nx.draw_networkx_edge_labels(G, pos=pos)

img_path = 'data\\semantic_graph_streets.jpg'
# Save the picture
plt.savefig(img_path)

print(f'Image is saved in {img_path}.')
'''

'''
df[text_column].dropna(inplace=True)
df[text_column] = df[text_column].astype(str)
df[["Street", "Score"]] = df[text_column].progress_apply(
    lambda t: g.extract_ner_street(t)
)
df["Street"] = df[[text_column, "Street"]].progress_apply(
    lambda row: g.get_ner_address_natasha(
        row, g.exceptions, text_column
    ),
    axis=1,
)
df = df[df.Street.notna()]
df = df[df["Street"].str.contains("[а-яА-Я]")]
'''