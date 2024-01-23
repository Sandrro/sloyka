from pathlib import Path


# Получаем строку, содержащую путь к рабочей директории:
dir_path = str(Path.cwd())

def get_dir_path(system_devider: str, exact_path: list) -> str:
        ### Функция объединяет путь на основе символа-разделителя

        tmp_path = str(dir_path).split(system_devider)
        tmp_path = tmp_path[:len(tmp_path)]
        joined_path = [j for i in [tmp_path, exact_path] for j in i]
        goal_dir_path = system_devider.join(joined_path)
        
        return goal_dir_path

def get_system_dir_path(goal_path: list, curren_path: str=dir_path) -> str:
        ### Функция определяет разделитель и возвращает целевой путь до дирректории

        if '/' in curren_path:
                # Linux & MacOS
                system_dir_path = get_dir_path('/', goal_path)

                return system_dir_path
        
        elif '\\' in curren_path:
                # Windows
                system_dir_path = get_dir_path('\\', goal_path)

                return system_dir_path

        else:
                print('Error, cannot define path devider.')


# Пути к папкам с данными и моделями
                
SEMANTIC_VALUE_DIR_RATH = get_system_dir_path(['semantic_vector'])

RAW_DATA_DIR_PATH = get_system_dir_path(['semantic_vector', 'data', 'train', 'raw'])

CLEAN_DATA_DIR_PATH = get_system_dir_path(['semantic_vector','data', 'train', 'clean'])

OUTPUT_DATA_DIR_PATH = get_system_dir_path(['semantic_vector','data', 'output'])

CUSTOM_MODELS_DIR_PATH = get_system_dir_path(['semantic_vector','models', 'custom'])

IMG_OUTPUT_DATA_DIR_PATH = get_system_dir_path(['img'])
