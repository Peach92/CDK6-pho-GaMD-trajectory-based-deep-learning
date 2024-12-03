import os
import mdtraj as md
from contact_map import ContactFrequency
import numpy as np
from PIL import Image
import random
import multiprocessing

def prep_data_system(tmp_dir):
    # traj_dir = './' + tmp_dir + '/nowat-md2/nowat4-md.nc'
    traj_dir = './' + tmp_dir + '/c1.pdb'
    top_dir  = './' + tmp_dir + '/comp.prmtop'

    print("traj_dir:",traj_dir)
    print("top:", top_dir)
    system = md.load(traj_dir,top=top_dir)
    len_system = len(system)
    print(len_system)
    image_list = []
    for i in np.arange(0, len_system):
        system_freq = ContactFrequency(system[i])
        system_freq = system_freq.residue_contacts
        system_freq = system_freq.df
        system_freq = system_freq.to_numpy()
        system_cmap = np.nan_to_num(system_freq)

        indices_one = (system_cmap == 1)
        indices_zero = (system_cmap == 0)
        system_cmap[indices_one] = 0
        system_cmap[indices_zero] = 1

        system_cmap = system_cmap * 255
        system_cmap = system_cmap.astype(np.uint8)

        system_img = Image.fromarray(system_cmap, 'L')
        image_list.append(system_img)

    # assert(len(image_list) == len_system)
    random.shuffle(image_list)

    img_dir = './test_img/train/' + tmp_dir
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f'folder {img_dir} create successfully.')
    else:
        print(f'folder {img_dir} exist.')

    img_valid_dir = './test_img/valid/' + tmp_dir
    if not os.path.exists(img_valid_dir):
        os.makedirs(img_valid_dir)
        print(f'folder {img_valid_dir} create successfully.')
    else:
        print(f'folder {img_valid_dir} exist.')

    img_file = './test_img/train/' + tmp_dir + '/' + 'train-1-1' + '.jpg'            
    image_list[0].save(img_file)


if __name__ == '__main__':
    # sys_dirs = ['brd4-h1b', 'brd4-jq1', 'brd4-tvu', 'brd9-h1b', 'brd9-jq1', 'brd9-tvu']
    # sys_dirs = ['brd4-h1b']
    sys_dirs = ['4ez5n', '4ez5p', '5l2sn', '5l2sp', '5l2tn', '5l2tp']


    pool = multiprocessing.Pool()
    pool.map(prep_data_system, sys_dirs)
    pool.close()
    pool.join()

