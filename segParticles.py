#%%
import argparse
import starfile
import os
parser = argparse.ArgumentParser(description='ParticleSeg: Segment your particles')
parser.add_argument('-t','--train',  dest='train',help='Train models with current dataset.')
parser.add_argument('-i','--infer',  help='Take a metadata(starfile) and return cleaned metadata.')
parser.add_argument('-r','--raw_data', help='Take raw data directory(where *.mrcs stored) if metadata for inference is provided.')
parser.add_argument('-d','--drop_ratio',  default=0.3, help='Ratio of the particles to be dropped')
parser.add_argument('-c','--continue', dest='continues', action='store_true',help='continue inference from current tmp files')
args = parser.parse_args()
base_models = [
  #'custom',
  'DenseNet121',
  'DenseNet169',
  'DenseNet201',
  'EfficientNetB0',
  'ResNet101'
]
output_folder = '/ssd/particleSeg/'
from inference import Inference_star, Inference_star_new
from train import Train
def train(model_name, num_to_use):
    model_Train = Train(model_name=model_name, num_to_use=num_to_use)
    model_Train.train()

if __name__ == "__main__":
    if args.train == 'all':
      for base_model in base_models:
        train(base_model, num_to_use=18000)
    elif args.train != None:
        train(args.train,  num_to_use=360000)
    if args.infer == 'default':
        star_file = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job011/particles.star'
        raw_data_dir = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job011/goodmrc_auto/'
        inference = Inference_star(star_file=star_file, raw_data_dir=raw_data_dir, ratio=args.drop_ratio, continues=args.continues)
    elif args.infer != None:
        #inference = Inference_star(star_file=args.infer, raw_data_dir=args.raw_data, ratio=args.drop_ratio, continues=args.continues)
        num_particles = len(starfile.read(args.infer))
        batch_size = 256
        num_batch = num_particles // batch_size + 1 
        if not args.continues:
            os.system(f'rm -rf {output_folder}')
            print('start a new run')
        inference = Inference_star_new(star_file=args.infer, raw_dir=args.raw_data, batch_size=batch_size)
        for i in range(num_batch):
            
            print(f'inferencing batch-{i} out of {num_batch}')
            inference.infer_batch(i)
        inference.concat()
        inference.drop(args.drop_ratio)
