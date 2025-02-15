from GestureDetector import GestureDetector
import json

with open('train_config.json', 'r', encoding='utf8') as train_config_file:
    config = json.loads(train_config_file.read())

detector = GestureDetector(config['model_type'], config['path_to_init_checkpoint'])
print('Старт обучения:\n\n')
detector.fit(config['path_to_data'], int(config['epochs']), config['final_checkpoint_name'], config['final_model_type'])