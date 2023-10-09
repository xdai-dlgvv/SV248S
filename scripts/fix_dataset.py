# this script is to fix the number error for image sequences
# normalize all sequences start from 000001.tiff

import os

dataset_root = '/data/SV248A10-SOT-20210707'   # sv248s dataset root directory

for scene in ['01', '02', '03', '04', '05', '06']:
    seqs_dir = os.path.join(dataset_root, scene, 'sequences')
    seqs = os.listdir(seqs_dir)

    for seq in seqs:
        if seq.isnumeric():
            seq_dir = os.path.join(seqs_dir, seq)
            all_images = [img for img in os.listdir(seq_dir) if img.endswith('.tiff')]
            all_images.sort()
            if all_images[0] == '000001.tiff':
                continue

            print('Find Target Sequence: %s' % seq_dir)

            for i, image in enumerate(all_images):
                os.rename(src=os.path.join(seq_dir, image),
                          dst=os.path.join(seq_dir, '%06d.tiff' % (i+1)))
print('DONE!')