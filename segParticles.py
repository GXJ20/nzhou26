'''
things to be fixed:
1. Change train class to an object-orient style

'''
import argparse
import starfile
import os
import pandas as pd
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
import inference 
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
    if args.infer != None:
        num_particles = len(starfile.read(args.infer))
        batch_size = 256
        num_batch = num_particles // batch_size + 1 
        if not args.continues:
            os.system(f'rm -rf {output_folder}')
            print('start a new run')
        infer = inference.Inference_star(star_file=args.infer, raw_dir=args.raw_data, batch_size=batch_size)
        ets_name_dict = []
        for i in range(num_batch):
            print(f'inferencing batch-{i+1} out of {num_batch}')
            ets_name_dict.append(infer.infer_batch(i))
        total_ets = pd.concat(ets_name_dict, axis=0, ignore_index=True)
        infer.drop(total_ets,args.drop_ratio)
