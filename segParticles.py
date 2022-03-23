
import argparse
import starfile
import pandas as pd
parser = argparse.ArgumentParser(description='ParticleSeg: Segment your particles')
parser.add_argument('-i','--input_star',  help='Take a metadata(starfile) and return cleaned metadata.')
parser.add_argument('-r','--raw_data', help='Take raw data directory(where *.mrcs stored) if metadata for inference is provided.')
parser.add_argument('-d','--drop_ratio',  default=0.3, help='Ratio of the particles to be dropped')
parser.add_argument('-c','--continue', dest='continues', action='store_true',help='continue inference from current tmp files')
args = parser.parse_args()

output_folder = '/ssd/particleSeg/'
import inference 
    
if __name__ == "__main__":
    num_particles = len(starfile.read(args.input_star))
    batch_size = 1024
    num_batch = num_particles // batch_size + 1 
    infer = inference.Inference_star(star_file=args.input_star, raw_dir=args.raw_data, batch_size=batch_size)
    ets_name_dict = []
    for i in range(num_batch):
        print(f'inferencing batch-{i+1} out of {num_batch}')
        ets_name_dict.append(infer.infer_batch(i))
    total_ets = pd.concat(ets_name_dict, axis=0, ignore_index=True)
    infer.drop(total_ets,args.drop_ratio)
