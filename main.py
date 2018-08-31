import cv2
from morphologie import morphologie
from detect_yarns import detectYarns
from correct_yarns import correctYarns
from detect_defects import detectDefects_wefts, detectDefects_warps
from create_bckgnd import create_7s_images
from help_functions import getStem
from stats_of_cloth import *
import platform
import time

# Just tick the steps that you want to compute
b_train_cnn = False
b_retrain_cnn = False

b_results_cnn = True
b_morphologie = True
b_detect_yarns = True
b_correct_yarns = True
b_detect_defects = True

#### DIRECTORIES AND OTHER SETTINGS ####

if platform.system() == 'Windows':
    harddrive = 'V:'
elif platform.system() == 'Linux':
    harddrive = '/images'
else:
    print('WHHHHHHAAAAAA')

plain_dirs = []

plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric2/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric3/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric4/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric7/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric8/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric21/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric25/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric26/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric1/')

#twill_dirs = []
#twill_dirs.append(harddrive + '/ITA/MA_weninger/twill_weave/Fabric12/')
#twill_dirs.append(harddrive + '/ITA/MA_weninger/twill_weave/Fabric20/')
#twill_dirs.append(harddrive + '/ITA/MA_weninger/twill_weave/Fabric22/')
#twill_dirs.append(harddrive + '/ITA/MA_weninger/twill_weave/Fabric23/')
#twill_dirs.append(harddrive + '/ITA/MA_weninger/twill_weave/Fabric27/')


# Choose between plain or twill
fabric_dirs = plain_dirs
#fabric_dirs = twill_dirs

dirs_groundtruth = [fabric_dir + 'Annotated50mm/' for fabric_dir in fabric_dirs]
model_paths = [fabric_dir + 'model_fcn_classic.h5' for fabric_dir in fabric_dirs]

dirs_test_images = [fabric_dir + 'test_images/' for fabric_dir in fabric_dirs]
target_dirs_fcn_result = [fabric_dir + 'result_fcn_classic/' for fabric_dir in fabric_dirs]
target_dirs_morphed = [fabric_dir + 'morphed_fcn_classic/' for fabric_dir in fabric_dirs]
target_dirs_yarns_detected = [fabric_dir + 'yarns_detected_fcn_classic/' for fabric_dir in fabric_dirs]
target_dirs_yarns_corrected = [fabric_dir + 'yarns_corrected_fcn_classic/' for fabric_dir in fabric_dirs]
target_dirs_defects_detected = [fabric_dir + 'defects_detected_fcn_classic/' for fabric_dir in fabric_dirs]
MCD_paths = [fabric_dir + 'MCD_fcn_classic' for fabric_dir in fabric_dirs]
plot_locs = [fabric_dir + 'result_fcn_classic.png' for fabric_dir in fabric_dirs]

if b_train_cnn:
    from train import train

    for idx, dir_groundtruth in enumerate(dirs_groundtruth[0:1]):
        print(dir_groundtruth)
        model_name = model_paths[idx]  # Where to store and load the CNN Model
        dirs_other_gts = dirs_groundtruth.copy()
        dirs_other_gts.remove(dir_groundtruth) #Every groundtruth dir, EXCEPT the current one, train only on other fabrics!

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

        #name = input("all necessary images??")
        train(model_name, imgs)
        input("Press Enter to continue...")

if b_retrain_cnn:
    from train import retrain

    imgs = []
    for idx, dir_groundtruth in enumerate(dirs_groundtruth):
        model_name = model_paths[idx]  # Where to store and load the CNN Model
        front_names = (sorted(glob.glob(os.path.join(dir_groundtruth, 'fl*.png'))))
        back_names = (sorted(glob.glob(os.path.join(dir_groundtruth, 'bl*.png'))))
        groundtruth_names = (sorted(glob.glob(os.path.join(dir_groundtruth, 'gt*.png'))))

        # Load all images, and bring them in the right format for training
        for front, back, groundtruth in zip(*(front_names, back_names, groundtruth_names)):
            print(front + " " + back + " " + groundtruth)
            fl_im = cv2.imread(front)
            bl_im = cv2.imread(back)
            gt_im = cv2.imread(groundtruth)
            imgs.append(create_7s_images(front=fl_im, back=bl_im, groundtruth=gt_im))

        print(front_names)

        retrain(model_name, imgs)


if b_results_cnn:
    import tensorflow as tf
    tf.python.control_flow_ops = tf
    import keras
    from Softmax2D import Softmax2D, categorical_accuracy_fcn
    from test import test, test2gpu, test3gpu

    for idx, model_name in enumerate(model_paths):
        dir_images = dirs_test_images[idx]
        target_dir_fcn_result = target_dirs_fcn_result[idx]
        print (model_name + "   " + dir_images + "    " + target_dir_fcn_result)

        with tf.device('/gpu:0'):
            model0 = keras.models.load_model(model_name, custom_objects={"Softmax2D": Softmax2D,  "categorical_accuracy_fcn": categorical_accuracy_fcn})

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
            print(target_dir_fcn_result + target_name)

            start1 = time.time()
            test(model=model0, img=img6s, target_name=target_dir_fcn_result + target_name)
            end1 = time.time()
            print(end1 - start1)
            times.append(end1-start1)
            if times.__len__() == 11:
                breakpoint=1
                times = times[1:]
                print('Dauer: ' + str(np.mean(times)))
                input("coucou")




#### MORPHOLOGICAL OPERATORS ####

if b_morphologie:
    for idx, target_dir_fcn_result in enumerate(target_dirs_fcn_result):
        target_dir_morphed = target_dirs_morphed[idx]
        images_names = glob.glob(os.path.join(target_dir_fcn_result, '*.png'))

        if not os.path.exists(target_dir_morphed):
            os.makedirs(target_dir_morphed)


        for img_name in images_names[152:]:
            im_number = getStem(img_name)[3:]
            morphologie(img_name=img_name, target_dir=target_dir_morphed, target_name=im_number)


#### DETECT YARNS ####

if b_detect_yarns:
    for idx, target_dir_morphed in enumerate(target_dirs_morphed):
        target_dir_yarns_detected = target_dirs_yarns_detected[idx]
        dir_images = dirs_test_images[idx]

        background_images = sorted(glob.glob(os.path.join(target_dirs_fcn_result[idx], '*.png')))
        hor_names = sorted(glob.glob(os.path.join(target_dir_morphed, 'h*.png')))
        ver_names = sorted(glob.glob(os.path.join(target_dir_morphed, 'v*.png')))

        if not os.path.exists(target_dir_yarns_detected):
            os.makedirs(target_dir_yarns_detected)

        for idx2, hor in enumerate(hor_names[152:]):
            ver = ver_names[idx2+152]
            fl_im = cv2.imread(background_images[idx2+152])
            target_name = target_dir_yarns_detected + 'd' + getStem(hor)[1:]
            detectYarns(hor_name=hor, ver_name=ver, target_name=target_name, write_images=False, image=fl_im)


#### CORRECT YARNS ####

if b_correct_yarns:
    for idx, target_dir_yarns_detected in enumerate(target_dirs_yarns_detected):
        target_dir_yarns_corrected = target_dirs_yarns_corrected[idx]
        dir_images = dirs_test_images[idx]

        background_images = sorted(glob.glob(os.path.join(target_dirs_fcn_result[idx], '*.png')))
        yarnsPickled_weft = glob.glob(os.path.join(target_dir_yarns_detected, '*_weft.p'))
        yarnsPickled_warp = glob.glob(os.path.join(target_dir_yarns_detected, '*_warp.p'))

        if not os.path.exists(target_dir_yarns_corrected):
            os.makedirs(target_dir_yarns_corrected)


        for idx2, pickled_path_weft in enumerate(yarnsPickled_weft[152:]):
            print("image: " + getStem(pickled_path_weft))
            pickled_path_warp = yarnsPickled_warp[idx2+152]
            target_name = target_dir_yarns_corrected + 'c' + getStem(pickled_path_weft)[1:-5]
            fl_im = cv2.imread(background_images[idx2+152])
            correctYarns(file_name_weft=pickled_path_weft, file_name_warp=pickled_path_warp, target_name=target_name, write_images=False, image=fl_im)


#### DETECT DEFECTS ####

if b_detect_defects:
    import matplotlib.patches as mpatches

    for idx, target_dir_yarns_corrected in enumerate(target_dirs_yarns_corrected):
        target_dir_defects_detected = target_dirs_defects_detected[idx]
        dir_images = target_dirs_fcn_result[idx]
        #dir_images = dirs_test_images[idx]
        MCD_path = MCD_paths[idx]
        plot_loc = plot_locs[idx]

        background_images = sorted(glob.glob(os.path.join(dir_images, '*.png')))
        #background_images = sorted(glob.glob(os.path.join(dir_images, 'fl*.png')))

        yarnsPickled_weft = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_weft.p')))
        yarnsPickled_warp = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_warp.p')))

        compute_MCD_weft(weftsPickled=yarnsPickled_weft, target_path=MCD_path + "_weft.p")
        compute_MCD_warp(warpsPickled=yarnsPickled_warp, target_path=MCD_path + "_warp.p")

        if not os.path.exists(target_dir_defects_detected):
            os.makedirs(target_dir_defects_detected)

        fault_counts = []

        robust_cov_warp = pickle.load(open(MCD_path + "_warp.p", "rb"))
        robust_cov_weft = pickle.load(open(MCD_path + "_weft.p", "rb"))

        for idx2, warp_path in enumerate(yarnsPickled_warp[152:]):
            weft_path = yarnsPickled_weft[idx2+152]
            target_name_warp = target_dir_defects_detected + getStem(warp_path)[1:]
            target_name_weft = target_dir_defects_detected + getStem(weft_path)[1:]
            fl_im = cv2.imread(background_images[idx2+152])
            dens00a, density01a, dens10a, dens11a = detectDefects_warps(file_name=warp_path, target_name=target_name_warp, robust_cov=robust_cov_warp, write_images=True, image=fl_im)
            dens00e, density01e, dens10e, dens11e = detectDefects_wefts(file_name=weft_path, target_name=target_name_weft, robust_cov=robust_cov_weft, write_images=True, image=fl_im)
            fault_counts.append(((dens00a + dens00e) / 2, (density01a + density01e) / 2, (dens10a + dens10e) / 2,
                                 (dens11a + dens11e) / 2))

        pickle.dump(fault_counts, open(target_dir_defects_detected + 'fault_counts.p', "wb"))

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
        x=3