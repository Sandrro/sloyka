import wget
import os
import zipfile


# model url
url = 'http://vectors.nlpl.eu/repository/20/220.zip'

# file name
name = '220.zip'

# path for semantic module dirrectory
current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))

# path for models
models_path = os.path.join(current_dir, 'models')

# path for saving downloaded model
file_path = os.path.join(models_path, name)


# downloading model
wget.download(url, file_path)

# extracting zip

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extract('model.bin', models_path)

# delliting donloading zip
if os.path.isfile(file_path): 
    os.remove(file_path) 
    print("Succesfully removed zip")

else: print("File doesn't exists!")

