#%%
import argparse
parser = argparse.ArgumentParser(description='ParticleSeg: Segment your particles')
parser.add_argument('--train', type=str, default=None, help='Train models with current dataset.')
parser.add_argument('--infer', type=str, default=None, help='Take a metadata(starfile) and return cleaned metadata.')
parser.add_argument('--raw_data', type=str, default=None, help='Take raw data directory if metadata for inference is provided.')
args = parser.parse_args()
base_models = [
  'custom',
  'DenseNet121',
  'DenseNet169',
  'DenseNet201',
  'EfficientNetB0',
  'ResNet101'
]

from inference import Inference_star
from train import Train
def train(model_name, num_to_use):
    model_Train = Train(model_name=model_name, num_to_use=num_to_use)
    model_Train.train()

def infer(star_file, raw_data_dir):
    inference = Inference_star(star_file=star_file, raw_data_dir=raw_data_dir)
    inference.e2e_infer()

if __name__ == "__main__":
    if args.train == 'all':
      for base_model in base_models:
        train(base_model, num_to_use=16000)
    elif args.train != None:
      train(args.train,  num_to_use=290000)
    if args.infer == 'default':
        infer(star_file='/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job011/particles.star'
        ,raw_data_dir='/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job011/goodmrc_auto/')
    elif args.infer != None:
        infer(star_file=args.infer, raw_data_dir=args.raw_data)
