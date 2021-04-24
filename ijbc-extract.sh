#!/bin/bash
python ijbc-extract.py --gpu 0 --config_path configs/arc_res50.yaml \
    --meta_path /s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJBC_backup.npz \
    --input_path /s/red/b/nobackup/data/portable/tbiom/models/face.evoLVe.PyTorch/IJBC_align_112x112 \
    --output_path ijbc_embs_arc_res50.npy
python ijbc-extract.py --gpu 0 --config_path configs/arc_mbv2.yaml \
    --meta_path /s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJBC_backup.npz \
    --input_path /s/red/b/nobackup/data/portable/tbiom/models/face.evoLVe.PyTorch/IJBC_align_112x112 \
    --output_path ijbc_embs_arc_mbv2.npy    