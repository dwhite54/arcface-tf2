import numpy as np
import os
import cv2
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import argparse
from tqdm.auto import tqdm

from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

def main(args):
    ijbc_meta = np.load(args.meta_path)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    #cfg = load_yaml('configs/arc_res50.yaml')
    cfg = load_yaml(args.config_path)

    model = ArcFaceModel(size=cfg['input_size'],
            backbone_type=cfg['backbone_type'],
            training=False)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    img_names = [os.path.join(args.input_path, img_name.split('/')[-1]) for img_name in ijbc_meta['img_names']]

    embedding_size = cfg['embd_shape']
    batch_size = cfg['batch_size']
    img_size = cfg['input_size']
    
    def read_img(filename):
        raw = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.cast(img, tf.float32)
        img = img / 255
        return img
    
    dataset = tf.data.Dataset.from_tensor_slices(img_names)
    dataset = dataset.map(read_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    embeddings = model.predict(dataset, batch_size=batch_size, verbose=1)
    
    print('embeddings', embeddings.shape)
    np.save(args.output_path, embeddings)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path to the target model's yaml config file",
                        type=str)
    parser.add_argument("--gpu", help="Which GPU to use for feature extraction",
                        type=str)
    parser.add_argument("--meta_path", help="The path to the IJBC_backup.npz meta file generated from InsightFace code.", type=str)
    parser.add_argument("--input_path", help="The path to aligned images named in the meta file", type=str)
    parser.add_argument("--output_path", help="The path to the numpy file containing output embeddings",
                        type=str, default='embeds.npy')
    args = parser.parse_args()
    print(args)
    if '.npy' not in args.output_path:
        print('output_path should contain .npy')
        exit(-1)
    main(args)