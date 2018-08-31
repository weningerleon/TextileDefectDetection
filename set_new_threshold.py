import cv2
from detect_defects import detectDefects_wefts, detectDefects_warps
from help_functions import getStem
from stats_of_cloth import *
import platform

plain_dirs = []


if platform.system() == 'Windows':
    harddrive = 'V:'
elif platform.system() == 'Linux':
    harddrive = '/images'
else:
    print('WHHHHHHAAAAAA')

plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric3/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric1/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric2/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric4/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric7/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric8/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric21/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric25/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric26/')

#U_net_small
#threshold1s = (500,100,30,300,1000,1000,30,30,30)
#threshold2s = (6,30,20,8,4.6,0.3,25,8,16)
# FCN
#threshold1s = (500,100,30,300,1000,3000,30,30,30)
#threshold2s = (1.9,27,10,3.8,3.3,0.3,23,1.1,36.9)
# U_net_big
threshold1s = (30,500,100,30,300,1000,1000,30,30,30)
threshold2s = (6, 1,9.5,6,6,3.6,5,11,1.5,10)

# Choose between plain or twill
fabric_dirs = plain_dirs
#fabric_dirs = twill_dirs

dirs_groundtruth = [fabric_dir + 'Annotated50mm/' for fabric_dir in fabric_dirs]

dirs_test_images = [fabric_dir + 'test_images/' for fabric_dir in fabric_dirs]
target_dirs_fcn_result = [fabric_dir + 'result/' for fabric_dir in fabric_dirs]
target_dirs_yarns_corrected = [fabric_dir + 'yarns_corrected/' for fabric_dir in fabric_dirs]
target_dirs_defects_detected = [fabric_dir + 'defects_detected/' for fabric_dir in fabric_dirs]
MCD_paths = [fabric_dir + 'MCD' for fabric_dir in fabric_dirs]
plot_locs = [fabric_dir + 'result' for fabric_dir in fabric_dirs]



import matplotlib.patches as mpatches
import matplotlib.lines as mlines

for idx, target_dir_yarns_corrected in enumerate(target_dirs_yarns_corrected):
    threshold1 = threshold1s[idx]
    threshold2 = threshold2s[idx]
    target_dir_defects_detected = target_dirs_defects_detected[idx]
    dir_images = target_dirs_fcn_result[idx]
    # dir_images = dirs_test_images[idx]
    MCD_path = MCD_paths[idx]
    plot_loc = plot_locs[idx]

    background_images = sorted(glob.glob(os.path.join(dir_images, '*.png')))
    # background_images = sorted(glob.glob(os.path.join(dir_images, 'fl*.png')))

    yarnsPickled_weft = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_weft.p')))
    yarnsPickled_warp = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_warp.p')))

    fault_counts = []

    robust_cov_warp = pickle.load(open(MCD_path + "_warp.p", "rb"))
    robust_cov_weft = pickle.load(open(MCD_path + "_weft.p", "rb"))

    for idx2, warp_path in enumerate(yarnsPickled_warp):
        weft_path = yarnsPickled_weft[idx2]
        target_name_warp = target_dir_defects_detected + getStem(warp_path)[1:]
        target_name_weft = target_dir_defects_detected + getStem(weft_path)[1:]
        fl_im = cv2.imread(background_images[idx2])
        dens00a, density01a, dens10a, dens11a = detectDefects_warps(file_name=warp_path, target_name=target_name_warp,
                                                                    robust_cov=robust_cov_warp, write_images=False,
                                                                    image=fl_im, threshold=threshold1)
        dens00e, density01e, dens10e, dens11e = detectDefects_wefts(file_name=weft_path, target_name=target_name_weft,
                                                                    robust_cov=robust_cov_weft, write_images=False,
                                                                    image=fl_im, threshold=threshold1)
        fault_counts.append(((dens00a + dens00e) / 2, (density01a + density01e) / 2, (dens10a + dens10e) / 2, (dens11a + dens11e) / 2))
        #print(str(fault_counts[idx2]))

    pickle.dump(fault_counts, open(target_dir_defects_detected + 'fault_counts.p', "wb"))

    tp = 0 # true positive
    fp = 0 # false positive
    tn = 0 # true negative
    fn = 0 # false negative
    for idx2, pickled_path in enumerate(yarnsPickled_weft):
        char_correct = getStem(pickled_path)[5]
        if idx2 == 52:
            char_correct = 'f'
        if char_correct == 'c':
            plt.plot(idx2, np.max(fault_counts[idx2]), color='g', marker='s', markersize=3, markeredgewidth=0)
            if np.max(fault_counts[idx2]) > threshold2:
                fp += 1
                #print("defect found: " + getStem(pickled_path))
            else:
                tn += 1
        elif char_correct == 'f':
            plt.plot(idx2, np.max(fault_counts[idx2]), color='r', marker='D', markersize=3, markeredgewidth=0)
            if np.max(fault_counts[idx2]) > threshold2:
                tp += 1
                #print("defect found: " + getStem(pickled_path))
            else:
                fn += 1
        else:
            plt.plot(idx2, sum(fault_counts[idx2]), 'b.')
            print('classification error')

    plt.axhline(y=threshold2, color='gray', linestyle='--')
    plt.xlabel('Image index')
    plt.ylabel('Proportion of defective float points')
    red_patch = mpatches.Patch(color='red', label='Faulty images')
    green_patch = mpatches.Patch(color='green', label='Fault-free images')
    gray_line = mlines.Line2D([],[],color='gray', label='Threshold 2', linestyle='dashed', linewidth=2)
    plt.legend(handles=[red_patch, green_patch, gray_line], loc=2)

    fa_no = fabric_dirs[idx][37:-1]
    plt.title('U-Net type network, Fabric ' + fa_no + '; Thresholds: ' + str(threshold1) + ' & ' + str(threshold2))

    print("Fabric: " + fa_no + "; tp=" + str(tp) + ", fp=" + str(fp) + " ,tn=" + str(tn) + ", fn=" + str(fn))
    plt.savefig(plot_loc + str(fa_no))
    #plt.show()
    plt.close()
    x = 3