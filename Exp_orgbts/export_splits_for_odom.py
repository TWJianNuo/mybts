import glob
import os
import numpy as np
import random
def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam):
    pose_imu2cam = np.eye(4)
    pose_imu2cam[0:3, 0:3] = np.reshape(imu2cam['R'], [3, 3])
    pose_imu2cam[0:3, 3] = imu2cam['T']

    pose_velo2cam = np.eye(4)
    pose_velo2cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
    pose_velo2cam[0:3, 3] = velo2cam['T']

    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = cam2cam['R_rect_00'].reshape(3, 3)

    intrinsic = np.eye(4)
    intrinsic[0:3, 0:3] = cam2cam['P_rect_02'].reshape(3, 4)[0:3, 0:3]

    org_intrinsic = np.eye(4)
    org_intrinsic[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    extrinsic_from_intrinsic = np.linalg.inv(intrinsic) @ org_intrinsic
    extrinsic_from_intrinsic[0:3, 0:3] = np.eye(3)

    extrinsic = extrinsic_from_intrinsic @ R_rect_00 @ pose_velo2cam @ pose_imu2cam

    return intrinsic.astype(np.float32), extrinsic.astype(np.float32)

train_seq = \
    ['00 2011_10_03_drive_0027 000000 004540',
     '01 2011_10_03_drive_0042 000000 001100',
     "02 2011_10_03_drive_0034 000000 004660",
     "03 2011_09_26_drive_0067 000000 000800",
     "04 2011_09_30_drive_0016 000000 000270",
     "05 2011_09_30_drive_0018 000000 002760",
     "06 2011_09_30_drive_0020 000000 001100",
     "07 2011_09_30_drive_0027 000000 001100",
     "08 2011_09_30_drive_0028 001100 005170"]

test_seq = ["09 2011_09_30_drive_0033 000000 001590",
            "10 2011_09_30_drive_0034 000000 001200"]

root_path = '/media/shengjie/disk1/data/Kitti'
gt_root = '/home/shengjie/Documents/Data/Kitti/semidense_gt'

entries = list()
seqmap = dict()
for seqm in train_seq:
    mapentry = dict()
    mapid, seqname, stid, enid = seqm.split(' ')
    allpngs = glob.glob(os.path.join(root_path, seqname[0:10], seqname + '_sync', 'image_02/data/*.png'))
    allpngs.sort()

    calib_dir = os.path.join(root_path, seqname[0:10])

    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
    intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
    for p in allpngs:
        idx = int(p.split('/')[-1].split('.')[0])

        rgb_path = os.path.join(seqname[0:10], seqname + '_sync', 'image_02/data/{}.png'.format(str(idx).zfill(10)))
        depth_path = os.path.join(seqname[0:10], seqname + '_sync', 'image_02/{}.png'.format(str(idx).zfill(10)))
        if os.path.exists(os.path.join(gt_root, depth_path)):
            entries.append("{} {} {}".format(rgb_path, depth_path, str(intrinsic[0,0])))
    random.shuffle(entries)
    with open('train_test_inputs/odom_train_files.txt', 'w') as f:
        for idx, e in enumerate(entries):
            if idx < len(entries) - 1:
                f.write(e + '\n')
            else:
                f.write(e)


entries = list()
seqmap = dict()
for seqm in test_seq:
    mapentry = dict()
    mapid, seqname, stid, enid = seqm.split(' ')
    allpngs = glob.glob(os.path.join(root_path, seqname[0:10], seqname + '_sync', 'image_02/data/*.png'))
    allpngs.sort()

    calib_dir = os.path.join(root_path, seqname[0:10])

    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
    intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
    for p in allpngs:
        idx = int(p.split('/')[-1].split('.')[0])

        rgb_path = os.path.join(seqname[0:10], seqname + '_sync', 'image_02/data/{}.png'.format(str(idx).zfill(10)))
        depth_path = os.path.join(seqname[0:10], seqname + '_sync', 'image_02/{}.png'.format(str(idx).zfill(10)))
        if os.path.exists(os.path.join(gt_root, depth_path)):
            entries.append("{} {} {}".format(rgb_path, depth_path, str(intrinsic[0,0])))
    random.shuffle(entries)
    entries = entries[0:800]
    with open('train_test_inputs/odom_test_files.txt', 'w') as f:
        for idx, e in enumerate(entries):
            if idx < len(entries) - 1:
                f.write(e + '\n')
            else:
                f.write(e)