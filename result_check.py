# %%
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
class check_result():
    def __init__(self, base_name, model_dir='../data_particleSeg/models/segmentation/'):
        self.model_file = base_name
        self.csv_file = f'{base_name}-history.csv'
        self.model_dir = model_dir
    def display_seg(self):
        history = pd.read_csv(self.csv_file)
        plt.plot(history['val_loss'], label='val_loss')
        plt.plot(history['loss'], label = 'loss')
        plt.plot(history['val_updated_mean_io_u_2'], label='val_mean_IOU')
        plt.plot(history['updated_mean_io_u_2'], label = 'mean_IOU')
        plt.legend()
        plt.show()
    def total_result(self):
        model_paths = pathlib.Path(self.model_dir)
        names = []
        ious = []
        for item in model_paths.glob('*.csv'):
            names.append(item.name.split('--')[2])
            ious.append(float(item.name.split('--')[0]))
        plt.plot(ious)
        plt.ylabel('IOU')
        plt.ylim(0,100)
        plt.xticks(ticks=range(0,len(names)),labels=names)
        for i in range(0,len(names)):
            print(f'{names[i]}: {ious[i]}')
base_name = '../data_particleSeg/models/segmentation/60.77--25600--DenseNet169--2021-12-23.h5'

check_result(base_name=base_name).total_result()

# %%
