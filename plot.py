import numpy as np
import csv
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use(['science', 'ieee'])
# plt.style.use(['science', 'no-latex'])


image_folder = '.'
curr_dir = '/home/jypark/stat_resnet18'
relu_dir = 'relu'
aespa_dir = 'aespa'
postact_dir = 'postact'
preact_dir = 'preact'

label1 = 'relu'
label2 = 'aespa'
label3 = 'postact'
label4 = 'preact'

# loss, train_acc, test_acc
plot_type = 'test_acc'

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the title


def read_csv(filename, idx):
    data_sim = [0]
    with open(filename) as f:
        reader = csv.DictReader(f)
        next(reader)  # skip header
        for r in reader:
            data_sim.append(float(r[idx]))
    data = np.asarray(data_sim)
    return data


def plot_fid_shade(axs, plot_type, label, idx):
    if label == 'relu':
        cl = 'tab:orange'
        label_name = 'ReLU'
        dir_name = os.path.join(relu_dir, 'stat.csv')
    elif label == 'aespa':
        cl = 'tab:blue'
        label_name = 'Basis-wise'
        dir_name = os.path.join(aespa_dir, 'stat.csv')
    elif label == 'preact':
        cl = 'tab:green'
        label_name = 'Pre-Act'
        dir_name = os.path.join(preact_dir, 'stat.csv')
    elif label == 'postact':
        cl = 'tab:red'
        label_name = 'Post-Act'
        dir_name = os.path.join(postact_dir, 'stat.csv')

    matrix = []
    for i in range(4):
        filename = os.path.join(curr_dir, dir_name.format(i+1, i+1))
        data = read_csv(filename, idx)
        matrix.append(data[1:])
    matrix = np.asarray(matrix)
    x = [i for i in range(matrix.shape[1])]

    means = np.mean(matrix, axis=0)
    errs = np.std(matrix, axis=0)
    axs.plot(x, means, c=cl, linewidth=2, label=label_name)
    axs.fill_between(x, means - errs, means + errs, alpha=0.35,
                     edgecolor='#1B2ACC', facecolor=cl, linewidth=0.1, antialiased=True, interpolate=True)

    return x


if __name__ == '__main__':

    if plot_type == 'loss':
        y_label = 'Loss Value'
        idx = 'train_loss'
    elif plot_type == 'train_acc':
        y_label = 'Train Accuracy'
        idx = 'train_acc'
    else:
        y_label = 'Test Accuracy'
        idx = 'test_acc'

    # fig = plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # fig.tight_layout(pad=0, w_pad=0.5)

# axs.grid(which='both')
x = plot_fid_shade(ax1, 'loss', label1, 'train_loss')
x = plot_fid_shade(ax1, 'loss', label2, 'train_loss')
x = plot_fid_shade(ax1, 'loss', label3, 'train_loss')
x = plot_fid_shade(ax1, 'loss', label4, 'train_loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss Value')
# ax1.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

x = plot_fid_shade(ax2, 'train_acc', label1, 'train_acc')
x = plot_fid_shade(ax2, 'train_acc', label2, 'train_acc')
x = plot_fid_shade(ax2, 'train_acc', label3, 'train_acc')
x = plot_fid_shade(ax2, 'train_acc', label4, 'train_acc')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Train Accuracy')
# ax2.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

x = plot_fid_shade(ax3, 'test_acc', label1, 'test_acc')
x = plot_fid_shade(ax3, 'test_acc', label2, 'test_acc')
x = plot_fid_shade(ax3, 'test_acc', label3, 'test_acc')
x = plot_fid_shade(ax3, 'test_acc', label4, 'test_acc')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Test Accuracy')
# ax3.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
handles, labels = ax1.get_legend_handles_labels()
fig.set_size_inches(18, 5)
fig.legend(handles, labels, loc='upper center', ncol=4)
# plt.figlegend( lines, labels, loc = 'upper center', ncol=5, labelspacing=0. )


# fig.tight_layout()
fig.savefig('{}/stat_{}.pdf'.format(
    image_folder, plot_type))
