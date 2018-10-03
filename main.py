import cv2
import time
import glob
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tensorflow as tf

from morphologie import morphologie
from detect_yarns import detectYarns
from correct_yarns import correctYarns
from detect_defects import detectDefects_wefts, detectDefects_warps
from create_bckgnd import create_7s_images
from help_functions import getStem
from stats_of_cloth import *
from test import test
from train import train

#%% Settings

# For CUDA enabled devices
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Set this to true and set the corresponding directories for training the network
b_train_cnn = False

b_results_cnn = True
b_morphologie = True
b_detect_yarns = True
b_correct_yarns = True
b_detect_defects = True

# Plot statistics over the whole fabric - not possible with only one example image
b_plot_statistics = False


#### DIRECTORIES ####

dirs_groundtruth = []
model_paths = ['example/net.h5']

dirs_test_images = ['example/orig_fabric']
target_dirs_fcn_result = ['example/fcn_result']
target_dirs_morphed = ['example/morphed']
target_dirs_yarns_detected = ['example/yarns_detected']
target_dirs_yarns_corrected = ['example/yarns_corrected']
target_dirs_defects_detected = ['example/defects_detected']
MCD_paths = ['example/mcd_model']
plot_locs = ['example/plot.png']


if b_train_cnn:

    for idx, dir_groundtruth in enumerate(dirs_groundtruth[0:1]):
        print(dir_groundtruth)
        model_name = model_paths[idx]  # Where to store and load the CNN Model
        dirs_other_gts = dirs_groundtruth.copy()
        dirs_other_gts.remove(dir_groundtruth) # Every groundtruth dir, EXCEPT the current one, train only on other fabrics!

        if not dirs_other_gts:
            print("Just 1 fabric type, impossible to train on others!")
            break

        front_names = []
        back_names = []
        groundtruth_names = []
        imgs = []
        # Get filenames of all images
        for dir in dirs_other_gts:
            front_names.extend(sorted(glob.glob(os.path.join(dir, 'fl*.png'))))
            back_names.extend(sorted(glob.glob(os.path.join(dir, 'bl*.png'))))
            groundtruth_names.extend(sorted(glob.glob(os.path.join(dir, 'gt*.png'))))

        # Load all images, and bring them in the right format for training
        for front, back, groundtruth in zip(*(front_names, back_names, groundtruth_names)):
            print(getStem(front) + " " + getStem(back) + " " + getStem(groundtruth))
            fl_im = cv2.imread(front)
            bl_im = cv2.imread(back)
            gt_im = cv2.imread(groundtruth)
            imgs.append(create_7s_images(front=fl_im, back=bl_im, groundtruth=gt_im))

        train(model_name, imgs)


if b_results_cnn:

    for idx, model_name in enumerate(model_paths[0:1]):
        dir_images = dirs_test_images[idx]
        target_dir_fcn_result = target_dirs_fcn_result[idx]
        print (model_name + "   " + dir_images + "    " + target_dir_fcn_result)

        model0 = tf.keras.models.load_model(model_name)

        front_names = sorted(glob.glob(os.path.join(dir_images, 'fl*.png')))
        back_names = sorted(glob.glob(os.path.join(dir_images, 'bl*.png')))

        if not os.path.exists(target_dir_fcn_result):
            os.makedirs(target_dir_fcn_result)

        times = []
        for idx2, front_name in enumerate(front_names[:]):
            front_img = cv2.imread(front_name)
            back_name = back_names[idx2]
            back_img = cv2.imread(back_name)
            print("front: " + getStem(front_name) + ", back: " + getStem(back_name))

            target_name = 'res' + getStem(front_name)[2:] + '.png'
            img6s = create_7s_images(front_img, back_img)

            start1 = time.time()
            with tf.device('device:GPU:0'):
                test(model=model0, img=img6s, target_name=os.path.join(target_dir_fcn_result, target_name))
            end1 = time.time()
            print(end1 - start1)
            times.append(end1-start1)
            if times.__len__() == 11:
                breakpoint=1
                times = times[1:]
                print('Dauer: ' + str(np.mean(times)))


#### MORPHOLOGICAL OPERATORS ####

if b_morphologie:
    for idx, target_dir_fcn_result in enumerate(target_dirs_fcn_result[0:1]):
        target_dir_morphed = target_dirs_morphed[idx]
        images_names = glob.glob(os.path.join(target_dir_fcn_result, '*.png'))

        if not os.path.exists(target_dir_morphed):
            os.makedirs(target_dir_morphed)

        for img_name in images_names:
            im_number = getStem(img_name)[3:]
            morphologie(img_name=img_name, target_dir=target_dir_morphed, target_name=im_number)


#### DETECT YARNS ####

if b_detect_yarns:
    for idx, target_dir_morphed in enumerate(target_dirs_morphed[0:1]):
        target_dir_yarns_detected = target_dirs_yarns_detected[idx]
        dir_images = dirs_test_images[idx]

        background_images = sorted(glob.glob(os.path.join(target_dirs_fcn_result[idx], '*.png')))
        hor_names = sorted(glob.glob(os.path.join(target_dir_morphed, 'h*.png')))
        ver_names = sorted(glob.glob(os.path.join(target_dir_morphed, 'v*.png')))

        if not os.path.exists(target_dir_yarns_detected):
            os.makedirs(target_dir_yarns_detected)

        for idx2, hor in enumerate(hor_names):
            ver = ver_names[idx2]
            fl_im = cv2.imread(background_images[idx2])
            target_name = os.path.join(target_dir_yarns_detected, 'd' + getStem(hor)[1:])
            detectYarns(hor_name=hor, ver_name=ver, target_name=target_name, write_images=False, image=fl_im)


#### CORRECT YARNS ####

if b_correct_yarns:
    for idx, target_dir_yarns_detected in enumerate(target_dirs_yarns_detected[0:1]):
        target_dir_yarns_corrected = target_dirs_yarns_corrected[idx]
        dir_images = dirs_test_images[idx]

        background_images = sorted(glob.glob(os.path.join(target_dirs_fcn_result[idx], '*.png')))
        yarnsPickled_weft = glob.glob(os.path.join(target_dir_yarns_detected, '*_weft.p'))
        yarnsPickled_warp = glob.glob(os.path.join(target_dir_yarns_detected, '*_warp.p'))

        if not os.path.exists(target_dir_yarns_corrected):
            os.makedirs(target_dir_yarns_corrected)

        for idx2, pickled_path_weft in enumerate(yarnsPickled_weft):
            print("image: " + getStem(pickled_path_weft))
            pickled_path_warp = yarnsPickled_warp[idx2]
            target_name = os.path.join(target_dir_yarns_corrected, 'c' + getStem(pickled_path_weft)[1:-5])
            fl_im = cv2.imread(background_images[idx2])
            correctYarns(file_name_weft=pickled_path_weft, file_name_warp=pickled_path_warp, target_name=target_name, write_images=False, image=fl_im)


#### DETECT DEFECTS ####

if b_detect_defects:

    for idx, target_dir_yarns_corrected in enumerate(target_dirs_yarns_corrected[0:1]):
        target_dir_defects_detected = target_dirs_defects_detected[idx]
        dir_images = target_dirs_fcn_result[idx]
        #dir_images = dirs_test_images[idx]
        MCD_path = MCD_paths[idx]
        plot_loc = plot_locs[idx]

        background_images = sorted(glob.glob(os.path.join(dir_images, '*.png')))

        yarnsPickled_weft = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_weft.p')))
        yarnsPickled_warp = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_warp.p')))

        compute_MCD_weft(weftsPickled=yarnsPickled_weft, target_path=MCD_path + "_weft.p")
        compute_MCD_warp(warpsPickled=yarnsPickled_warp, target_path=MCD_path + "_warp.p")

        if not os.path.exists(target_dir_defects_detected):
            os.makedirs(target_dir_defects_detected)

        fault_counts = []

        robust_cov_warp = pickle.load(open(MCD_path + "_warp.p", "rb"))
        robust_cov_weft = pickle.load(open(MCD_path + "_weft.p", "rb"))

        for idx2, warp_path in enumerate(yarnsPickled_warp):
            weft_path = yarnsPickled_weft[idx2]
            target_name_warp = os.path.join(target_dir_defects_detected, getStem(warp_path)[1:])
            target_name_weft = os.path.join(target_dir_defects_detected, getStem(weft_path)[1:])
            fl_im = cv2.imread(background_images[idx2])
            dens00a, density01a, dens10a, dens11a = detectDefects_warps(file_name=warp_path, target_name=target_name_warp, robust_cov=robust_cov_warp, write_images=True, image=fl_im)
            dens00e, density01e, dens10e, dens11e = detectDefects_wefts(file_name=weft_path, target_name=target_name_weft, robust_cov=robust_cov_weft, write_images=True, image=fl_im)
            fault_counts.append(((dens00a + dens00e) / 2, (density01a + density01e) / 2, (dens10a + dens10e) / 2,
                                 (dens11a + dens11e) / 2))

        pickle.dump(fault_counts, open(os.path.join(target_dir_defects_detected, 'fault_counts.p'), "wb"))

if b_plot_statistics:
        threshold2 = 2
        tp = 0  # true positive
        fp = 0  # false positive
        tn = 0  # true negative
        fn = 0  # false negative
        for idx2, pickled_path in enumerate(yarnsPickled_weft):
            char_correct = getStem(pickled_path)[5]
            if char_correct == 'c':
                plt.plot(idx2, np.max(fault_counts[idx2]), 'g.')
                if np.max(fault_counts[idx2]) > threshold2:
                    fp += 1
                else:
                    tn += 1
            elif char_correct == 'f':
                plt.plot(idx2, np.max(fault_counts[idx2]), 'r.')
                if np.max(fault_counts[idx2]) > threshold2:
                    tp += 1
                else:
                    fn += 1
            else:
                plt.plot(idx2, sum(fault_counts[idx2]), 'b.')
                print('classification error')

        plt.xlabel('Image index')
        plt.ylabel('Fault count')
        red_patch = mpatches.Patch(color='red', label='Faulty images')
        green_patch = mpatches.Patch(color='green', label='Fault-free images')
        plt.legend(handles=[red_patch, green_patch], loc=2)

        plt.title('Fault count for all images of fabric 3')
        plt.savefig(plot_loc)
        plt.close()