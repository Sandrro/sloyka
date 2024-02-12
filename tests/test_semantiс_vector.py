import os
from gensim.models import word2vec

from factfinder.src.semantic_model import *


def test_initialize_save_model():
    '''This function tests creation od Word2Vec model and results of saving it to a file.'''
    
    result = []
    m = Semantic_model('data\\raw\\posts_spb_today.csv', 'text')
    # Model Initiation
    model = m.make_model()
    similarity = model.wv.similarity('полицейский', 'тротуар')
    train_check = 0 <= similarity <= 1
    result.append(train_check)
    # Saving the model
    model_path = 'semantic_models\\model.model'
    m.save_model(model=model, model_path=model_path)
    save_check = os.path.exists(model_path)
    result.append(save_check)

    assert result == [True, True]

def test_train_save_model():
    '''This function tests training of existing Word2Vec model and results of saving it to a file.'''
    
    result = []
    model = word2vec.Word2Vec.load('semantic_models\\model.model')
    old_similarity = model.wv.similarity('полицейский', 'тротуар')
    m = Semantic_model(file_path='data\\raw\\total_reports.csv')
    # Training an existing model on new data
    trained_model = m.train_model(model_path='semantic_models\\model.model', training_data_path='data\\raw\\total_reports.csv', column='Текст')
    similarity = trained_model.wv.similarity('полицейский', 'тротуар')
    train_check = similarity != old_similarity
    result.append(train_check)

    # Preservation of the pre-trained model
    m.save_model(model=trained_model, model_path='semantic_models\\trained_model.model')
    trained_model_path = 'semantic_models\\trained_model.model'
    path_check = os.path.exists(trained_model_path)
    result.append(path_check)
    
    assert result == [True, True]

def test_visualize_graph():
    '''This function tests creation of a graph and results of saving it to a file.'''
    
    img_path = 'data/kg1.jpg'
    # Construct a knowledge graph from a specified word or list of words
    Visualiztor(word='тротуар', model_path='semantic_models\\trained_model.model', depth=2, topn=10).save_graph_img(img_path=img_path)
    img_check = os.path.exists(img_path)

    assert img_check == True
    
