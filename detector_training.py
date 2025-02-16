import json
import os

from GestureDetector import GestureDetector


def detector_training():
    """
    The function starts training the model. This function has no arguments - it reads all
    the necessary settings from the config-file, which is generated in MotionControlApp.py.
    """
    with open(os.path.join('configs', 'train_config.json'), 'r', encoding='utf8') as train_config_file:
        config = json.loads(train_config_file.read())
    detector = GestureDetector(config['model_type'], config['path_to_init_checkpoint'])
    print('Старт обучения:\n\n')
    detector.fit(config['path_to_data'], int(config['epochs']), config['path_to_final_checkpoint'])
    print(f'Обучение завершено. Новый чекпойнт сохранён по указанному пути.\n')
    input('Нажмите Enter для завершения обучения.')


detector_training()