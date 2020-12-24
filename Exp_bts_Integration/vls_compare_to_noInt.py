import os
import PIL.Image as Image
import numpy as np

vlsrootint = '/media/shengjie/disk1/visualization/btspred/bts_intSDnetls1e6'
vlsrootNoint = '/media/shengjie/disk1/visualization/btspred/bts_121_org_bz4_cp'
vlsrootCp = '/media/shengjie/disk1/visualization/btspred/intvsNoint'
os.makedirs(vlsrootCp, exist_ok=True)

for i in range(652):
    img1 = Image.open(os.path.join(vlsrootint, str(i).zfill(6) + '.png'))
    w, h = img1.size
    img1 = np.array(img1)[int(h / 5 * 3)::, :]
    img2 = Image.open(os.path.join(vlsrootNoint, str(i).zfill(6) + '.png'))

    imgcombined = np.concatenate([img1, np.array(img2)], axis=0)
    Image.fromarray(imgcombined).save(os.path.join(vlsrootCp, str(i).zfill(6) + '.png'))