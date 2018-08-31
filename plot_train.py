import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy
from scipy.stats.mstats import normaltest
import matplotlib.patches as mpatches
import numpy as np

accloggers = []

accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric1\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric2\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric3\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric4\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric7\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric8\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric21\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric25\model_new_unet_complete_logger.p', "rb")))
accloggers.append(pickle.load(open(r'V:\ITA\MA_weninger\plain_weave\Fabric26\model_new_unet_complete_logger.p', "rb")))

labels = []
labels.append('Without Fabric 1')
labels.append('Without Fabric 2')
labels.append('Without Fabric 3')
labels.append('Without Fabric 4')
labels.append('Without Fabric 7')
labels.append('Without Fabric 8')
labels.append('Without Fabric 21')
labels.append('Without Fabric 25')
labels.append('Without Fabric 26')


NUM_COLORS = 10

cm = plt.get_cmap('gist_rainbow')

fig = plt.figure(0)
ax = fig.add_subplot(111)
axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 1])
ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

handles = []
for idx, acclogger in enumerate(accloggers):

    train_acc = acclogger[0]
    train_acc.insert(0,0.33)

    train_loss = acclogger[1]
    epoch_ends = acclogger[2]
    val_accs = acclogger[3]
    val_accs.insert(0,0.33)

    x = np.linspace(0,epoch_ends.__len__(),num=train_acc.__len__())



    p1 = ax.plot(x, train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Set Accuracy')

    handles.append(mpatches.Patch(color=p1[0].get_color(), label=labels[idx]))
    plt.legend(handles=handles, loc=4)


    plt.xticks(np.arange(0, 3.001, 1))

    fig = plt.figure(1)

    #plt.plot(x, train_acc)
    p2 = plt.plot(val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Set Accuracy')

    plt.legend(handles=handles, loc=4)


    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([0, 1])
    axes.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    plt.xticks(np.arange(0, 3.001, 1))

    x=3

plt.show()

red_patch = mpatches.Patch(color='m', label='Without Fabric 26')
red_patch = mpatches.Patch(color='y', label='Without Fabric 26')
red_patch = mpatches.Patch(color='k', label='Without Fabric 26')
red_patch = mpatches.Patch(color='c', label='Without Fabric 26')
red_patch = mpatches.Patch(color='c', label='Without Fabric 26')
