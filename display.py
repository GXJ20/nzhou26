# %%
import pathlib
import starfile
import random
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from particle import particles
import pandas as pd
import inference
seg_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/'
img_size = (256,256)


class plot_models():
    def __init__(self, model_dir, pattern):
        model_paths = list(pathlib.Path(model_dir).glob(pattern))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        model_metrics = [float(item.name.split('--')[0]) for item in model_paths]
        model_names =  ['@'.join(item.name.split('--')[2:4]).split(".")[0] for item in model_paths]
        
        self.display(model_metrics, model_names)
    def display(self, metrics, names):
        short_names = [f'{name[:10]}...' for name in names]
        plt.figure(figsize=(5,5))
        plt.plot(metrics)
        plt.ylim(0,100)
        plt.xticks(ticks=range(0,len(names)),labels=short_names, rotation=85)
        plt.show()
        for i in range(0,len(names)):
            print(f'{names[i]}: {metrics[i]}')

def display_infer_particles(star_file, raw_dir, num_to_display ,quality, seg_model_key='*.h5'):
    raw_imgs = []
    lp_imgs = []
    imgs = []
    metadata = starfile.read(star_file)
    model_paths =  list(pathlib.Path(seg_model_dir).glob(seg_model_key))
    model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
    seg_model = model_paths[0]
    if quality == 'good':
        metadata = metadata[metadata['rlnGroupName'] != 'group_111']
    elif quality == 'bad':
        metadata = metadata[metadata['rlnGroupName'] == 'group_111']
    for i in range(num_to_display):
        #random.seed(i)
        idx = random.randint(0, len(metadata))
        raw_img,img_name = inference.raw_from_star(idx, metadata, raw_dir)
        raw_imgs.append(raw_img)
        lp_imgs.append(gaussian_filter(raw_img, sigma=5))
        img = cv2.resize(raw_img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
        img = np.stack((img,)*3, axis=-1)
        imgs.append(img)
    width = round(np.sqrt(num_to_display))
    display_batch(raw_imgs, width)
    display_batch(lp_imgs, width)
    pred_batch = np.asarray(imgs)
    print(pred_batch.shape)
    pred = inference.segment(pred_batch,seg_model_name=seg_model)
    display_batch(pred, width)
    ratings = inference.ets(pred)
    display_batch(pred,width, titles=ratings)

def explain_ets(seg):
    plt.figure(figsize=(3,3))
    plt.imshow(seg, cmap='gray')
    unique, counts = np.unique(seg, return_counts=True)
    my_dict = dict(zip(unique, counts))
    ets = my_dict[2]/my_dict[0]
    print(my_dict)
    print(ets)
class check_seg_history():
    def __init__(self, csv_file):
        history = pd.read_csv(csv_file)
        plt.plot(history.iloc[:,1], label='val_loss')
        plt.plot(history.iloc[:,2], label = 'loss')
        plt.plot(history.iloc[:,3], label='val_mean_IOU')
        plt.plot(history.iloc[:,4], label = 'mean_IOU')
        plt.legend()
        plt.show()
class check_augment():
    def __init__(self, num_to_display) -> None:
        data_paths = pathlib.Path(data_dir)
        input_img_paths = list(data_paths.glob("*/raw/*"))
        input_img_paths = sorted(input_img_paths, key=os.path.basename)
        target_img_paths = list(data_paths.glob("*/label/*"))
        target_img_paths = sorted(target_img_paths, key=os.path.basename)
        random.Random(1337).shuffle(input_img_paths)
        random.Random(1337).shuffle(target_img_paths)
        input_img_paths = input_img_paths[:num_to_display]
        target_img_paths = target_img_paths[:num_to_display]
        aug_gen = particles(num_to_display, img_size, input_img_paths, target_img_paths,fold=1)
        no_aug_gen = particles(num_to_display, img_size, input_img_paths, target_img_paths,fold=0)
        self.width = round(np.sqrt(num_to_display))
        no_aug_batch_image, no_aug_batch_target = no_aug_gen.__getitem__(0)
        batch_image, batch_target = aug_gen.__getitem__(0)
        lp_no_aug_batch_image = [gaussian_filter(img,sigma=5) for img in no_aug_batch_image]

        display_batch(no_aug_batch_image)
        display_batch(lp_no_aug_batch_image)
        display_batch(no_aug_batch_target)
        display_batch(batch_image)
        display_batch(batch_target)
def display_test_particles(dataset='*', seg_model_key='*.h5',num_to_display=9):
    raw_img_paths = list(pathlib.Path(data_dir).glob(f'{dataset}/raw/*.npy'))
    label_img_paths = list(pathlib.Path(data_dir).glob(f'{dataset}/label/*.npy'))
    raw_img_paths = sorted(raw_img_paths, key=os.path.basename)
    label_img_paths = sorted(label_img_paths, key=os.path.basename)
    raw_imgs = []
    imgs = []
    label_imgs = []
    lp_imgs = []
    for i in range(num_to_display):
        #random.seed(i)
        idx = random.randint(0, len(raw_img_paths))
        img = np.load(raw_img_paths[idx])
        label_imgs.append(np.load(label_img_paths[idx]))
        raw_imgs.append(img)
        lp_imgs.append(gaussian_filter(img, sigma=5))
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
        img = np.stack((img,)*3, axis=-1)
        imgs.append(img)
    width = round(np.sqrt(num_to_display))
    display_batch(raw_imgs,width)
    display_batch(lp_imgs,width)
    display_batch(label_imgs,width)
    model_paths =  list(pathlib.Path(seg_model_dir).glob(seg_model_key))
    model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
    seg_model = model_paths[0]
    pred_batch = np.asarray(imgs)
    print(f'Using model: {seg_model}')
    pred = inference.segment(pred_batch,seg_model_name=seg_model)
    display_batch(pred, width)
    ratings = inference.ets(pred)
    display_batch(pred,width, titles=ratings)

class Plot_Picking():
    def __init__(self, star_file, mrc_dir) -> None:
        self.metadata = starfile.read(star_file)
        self.mrc_paths = list(pathlib.Path(mrc_dir).glob("*.mrc"))
    def pie_chart(self):
        labels = ['good', 'bad']
        num_good = len(self.metadata[self.metadata['rlnGroupName'] != 'group_111'])
        num_bad = len(self.metadata) - num_good
        sizes = [num_good, num_bad]
        explode = (0,0.1)
        fig1,ax1 = plt.subplots()
        ax1.pie(sizes,explode=explode, colors=['springgreen', 'orangered'], labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
        ax1.axis('equal')
        print(f'bad particles: {num_bad}')
        print(f'good particles: {num_good}')
        plt.show()
    def pick_on_mrc(self,idx):    
        star_name = self.mrc_paths[idx].name
        print(star_name[:-4])
        with mrcfile.open(self.mrc_paths[idx]) as mrc:
            mrc_data = gaussian_filter(mrc.data, sigma=7)
        df = self.metadata[self.metadata['rlnMicrographName'].str.contains(star_name)]
        df_right = df[df['rlnGroupName'] != 'group_111']
        df_wrong = df[df['rlnGroupName'] == 'group_111']
        fig, ax = plt.subplots(1,figsize=(10,15))
        ax.imshow(mrc_data, cmap='gray')
        ax.scatter(x=df_right['rlnCoordinateX'], y= df_right['rlnCoordinateY'], s= 250, facecolors='none', edgecolors='springgreen', linestyle='-')
        ax.scatter(x=df_wrong['rlnCoordinateX'], y= df_wrong['rlnCoordinateY'], s= 250, facecolors='none', edgecolors='orangered', linestyle='-')
        ax.invert_yaxis()
        ax.axis('off')
        print(star_name)
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)
class display_train_data():
    def __init__(self, data_dir, dataset='20s_10025', num_to_display=9):
        self.data_dir = data_dir
        self.dataset = dataset
        raw_img_paths = list(pathlib.Path(data_dir).glob(f'{dataset}/raw/*.npy'))
        label_img_paths = list(pathlib.Path(data_dir).glob(f'{dataset}/label/*.npy'))
        raw_img_paths = sorted(raw_img_paths, key=os.path.basename)
        label_img_paths = sorted(label_img_paths, key=os.path.basename)
        raw_imgs = []
        lp_imgs = []
        label_imgs = []
        for i in range(num_to_display):
            random.seed(i)
            idx = random.randint(0, len(raw_img_paths))
            raw_imgs.append(np.load(raw_img_paths[idx]))
            label_imgs.append(np.load(label_img_paths[idx]))
            lp_imgs = [gaussian_filter(img, sigma=5) for img in raw_imgs]
        width = round(np.sqrt(num_to_display))
        display_batch(raw_imgs,width)
        display_batch(lp_imgs,width)
        display_batch(label_imgs,width)
        print(label_imgs[0])
class Analysis_ETS():
    def __init__(self, star_file, pred_csv) -> None:
        self.metadata = starfile.read(star_file)
        self.pred_scores = pd.read_csv(pred_csv,index_col=0)
        
    def dispay_good_bad_distribution(self, quality='good'):
        good_particles = self.metadata[self.metadata['rlnGroupName'] != 'group_111']
        good_particles = good_particles['rlnImageName'].astype(str).values.tolist()
        plt.xlim(0,0.25)
        plt.xlabel('Edge to Signal Ratio')
        if quality == 'good':
            ets_list = self.pred_scores[self.pred_scores['name'].isin(good_particles)]
            hist_color = 'springgreen'
        elif quality == 'bad':
            ets_list = self.pred_scores[~self.pred_scores['name'].isin(good_particles)]
            hist_color = 'orangered'
        ets_list = ets_list['ets_ratio'].astype(float).values.tolist()
        n, bins, patches = plt.hist(ets_list, 20, facecolor=hist_color, alpha=0.7)
        plt.show()
    
    def ets_vs_defocus(self):
        total_defocus = self.metadata['rlnDefocusV'].astype(int).values.tolist()
        total_ets = self.pred_scores['ets_ratio'].astype(float).values.tolist()
        df = pd.DataFrame({
            'total_defocus':total_defocus,
            'total_ets':total_ets
        })
        plt.xlabel('particle defocus')
        plt.ylabel('particle ETS')
        plt.scatter(df['total_defocus'],df['total_ets'])
def display_batch(batch, width,titles=[]):
    plt.figure(figsize=(5,5))
    for i in range(1, width*width +1):
        plt.subplot(width, width, i)
        plt.imshow(batch[i-1], cmap='gray')
        if titles:
            plt.title(titles[i-1])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

star_file = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job029/particles.star'
raw_dir = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job029/goodmrc_auto/'
data_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/data_for_training/'
pred_csv='/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job029/IOU67.58--dropped0.3--backup.csv'
mrc_dir = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/CtfFind/job007/goodmrc_auto'

if __name__ =='__main__':
    display_test_particles(seg_model_key='*.h5', num_to_display=16)

    #analysis = Analysis_ETS(star_file=star_file, pred_csv=pred_csv)
    #analysis.dispay_good_bad_distribution('bad')
    #analysis.dispay_good_bad_distribution('good')
    #analysis.ets_vs_defocus()
    
    # coord_plot = Plot_Picking(star_file=star_file, mrc_dir=mrc_dir)
    # coord_plot.pie_chart()
    # coord_plot.pick_on_mrc(20)
    #display_infer_particles(star_file, raw_dir, 9, quality='all', seg_model_key='67.58*.h5')
    #display.explain_ets()
    #plot_models(seg_model_dir, '*290000*.csv')
    #check_seg_history('/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/66.32--290000--DenseNet169--2021-12-28.h5-history.csv')
    #check_augment(9)
    #display_train_data(data_dir=data_dir, dataset='empiar_10255')
# %%
