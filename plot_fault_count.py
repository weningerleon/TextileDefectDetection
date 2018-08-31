import matplotlib.patches as mpatches
from help_functions import getStem
from stats_of_cloth import *
import platform

#### DIRECTORIES AND OTHER SETTINGS ####

if platform.system() == 'Windows':
    harddrive = 'V:'
elif platform.system() == 'Linux':
    harddrive = '/images'
else:
    print('WHHHHHHAAAAAA')

plain_dirs = []
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric1/')
plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric2/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric3/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric4/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric7/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric8/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric21/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric25/')
#plain_dirs.append(harddrive + '/ITA/MA_weninger/plain_weave/Fabric26/')
fabric_dirs = plain_dirs

dirs_test_images = [fabric_dir + 'test_images/' for fabric_dir in fabric_dirs]
target_dirs_fcn_result = [fabric_dir + 'result_fcn_re/' for fabric_dir in fabric_dirs]
target_dirs_yarns_corrected = [fabric_dir + 'yarns_corrected/' for fabric_dir in fabric_dirs]
target_dirs_defects_detected = [fabric_dir + 'defects_detected_unet_small/' for fabric_dir in fabric_dirs]
plot_locs = [fabric_dir + 'result.png' for fabric_dir in fabric_dirs]


for idx, target_dir_yarns_corrected in enumerate(target_dirs_yarns_corrected):
    target_dir_defects_detected = target_dirs_defects_detected[idx]
    plot_loc = plot_locs[idx]

    yarnsPickled_weft = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_weft.p')))
    yarnsPickled_warp = sorted(glob.glob(os.path.join(target_dir_yarns_corrected, '*_warp.p')))

    fault_counts = pickle.load(open(target_dir_defects_detected + 'fault_counts.p', "rb"))

    if plain_dirs[idx][-2] == '3':
        fault_counts = fault_counts[:162]
        yarnsPickled_weft = yarnsPickled_weft[:162]
    threshold2 = 1
    tp = 0 # true positive
    fp = 0 # false positive
    tn = 0 # true negative
    fn = 0 # false negative
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

    #axes = plt.gca()
    #axes.set_ylim([0, 100])
    plt.xlabel('Image index')
    plt.ylabel('Fault count')
    red_patch = mpatches.Patch(color='red', label='Faulty images')
    green_patch = mpatches.Patch(color='green', label='Fault-free images')
    plt.legend(handles=[red_patch, green_patch], loc=2)

    plt.title('Fault count for all images of fabric ' + plain_dirs[idx][-3:-1])
    print("tp=" + str(tp) + ", fp=" + str(fp) + ", fn=" + str(fn) + " ,tn=" + str(tn))
    plt.show()
    #plt.savefig(plot_loc)
    plt.close()