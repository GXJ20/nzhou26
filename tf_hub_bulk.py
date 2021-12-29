# %%
import os
import sys
import pathlib
model_handle_map = {
  "efficientnetv2-s": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
  "efficientnetv2-m": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
  "efficientnetv2-l": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
  "efficientnetv2-s-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
  "efficientnetv2-m-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
  "efficientnetv2-l-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
  "efficientnetv2-xl-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
  "efficientnetv2-b0-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
  "efficientnetv2-b1-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
  "efficientnetv2-b2-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
  "efficientnetv2-b3-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
  "efficientnetv2-s-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
  "efficientnetv2-m-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
  "efficientnetv2-l-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
  "efficientnetv2-xl-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
  "efficientnetv2-b0-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
  "efficientnetv2-b1-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
  "efficientnetv2-b2-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
  "efficientnetv2-b3-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
  "efficientnetv2-b0": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
  "efficientnetv2-b1": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
  "efficientnetv2-b2": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
  "efficientnetv2-b3": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
  "efficientnet_b0": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b0/feature-vector/1",
  "efficientnet_b1": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b1/feature-vector/1",
  "efficientnet_b2": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b2/feature-vector/1",
  "efficientnet_b3": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b3/feature-vector/1",
  "efficientnet_b4": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b4/feature-vector/1",
  "efficientnet_b5": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b5/feature-vector/1",
  "efficientnet_b6": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b6/feature-vector/1",
  "efficientnet_b7": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b7/feature-vector/1",
  "bit_s-r50x1": "https://hub.tensorflow.google.cn/google/bit/s-r50x1/1",
  "inception_v3": "https://hub.tensorflow.google.cn/google/imagenet/inception_v3/feature-vector/4",
  "inception_resnet_v2": "https://hub.tensorflow.google.cn/google/imagenet/inception_resnet_v2/feature-vector/4",
  "resnet_v1_50": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v1_50/feature-vector/4",
  "resnet_v1_101": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v1_101/feature-vector/4",
  "resnet_v1_152": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v1_152/feature-vector/4",
  "resnet_v2_50": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature-vector/4",
  "resnet_v2_101": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_101/feature-vector/4",
  "resnet_v2_152": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_152/feature-vector/4",
  "nasnet_large": "https://hub.tensorflow.google.cn/google/imagenet/nasnet_large/feature_vector/4",
  "nasnet_mobile": "https://hub.tensorflow.google.cn/google/imagenet/nasnet_mobile/feature_vector/4",
  "pnasnet_large": "https://hub.tensorflow.google.cn/google/imagenet/pnasnet_large/feature_vector/4",
  "mobilenet_v2_100_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
  "mobilenet_v2_130_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
  "mobilenet_v2_140_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
  "mobilenet_v3_small_100_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
  "mobilenet_v3_small_075_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
  "mobilenet_v3_large_100_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
  "mobilenet_v3_large_075_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}
model_names = []
for item in pathlib.Path('models').glob("*pred_rating*"):
    model_name = item.name.split('rating_')[1].split('_2021')[0]
    model_names.append(model_name)
py_interpretor = '/home/zhou_ningkun/.conda/envs/work_env/bin/python'
for item in model_handle_map:
    if item in model_names:
      print(f"{item} already tested")
      continue
    try:
        command = f"{py_interpretor} pred_rating_train.py {item}"
        print(command)
        os.system(command)
    except(KeyboardInterrupt):
        sys.exit()
# %%
