import os
import gensim


# static model path
current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
models_dir = os.path.join(current_dir, 'models')
model_path = os.path.join(models_dir, 'model.bin')

# open static model
model_ru = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# model check
print(model_ru.similarity('снег_NOUN', 'тротуар_NOUN'))
