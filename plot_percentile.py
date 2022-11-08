import numpy as np
import csv
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use(['science', 'ieee'])
# plt.style.use(['science', 'no-latex'])


image_folder = '.'
curr_dir = '/home/jypark/stat_resnet18'
relu_dir = 'lr_cos/relu'
aespa_dir = 'lr_cos/aespa'
postact_dir = 'lr_cos/postact'
preact_dir = 'lr_cos/preact'

label1 = '1'
label2 = '20'
label3 = '80'
label4 = '99'

# loss, train_acc, test_acc
plot_type = 'test_acc'
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the title


def read_csv(filename, idx):
    data_sim = []
    with open(filename) as f:
        reader = csv.reader(f)
        # next(reader)  # skip header
        for r in reader:
            data_sim.append(float(r[idx]))
    data = np.asarray(data_sim)
    data = data.reshape((200, 491))
    # data = np.delete(data, np.s_[391:491], axis=1)
    # data = np.delete(data, np.s_[0:198], axis=0)
    average = np.mean(data, axis=1)
    max = np.max(data)
    print("max: ", max)
    return average


def plot_fid_shade(axs, plot_dir, label, idx):
    if label == '1':
        cl = 'tab:orange'
        label_name = '1th'
        dir_name = os.path.join(plot_dir, 'input_layer3.1.conv1_' + '1th.csv')
    elif label == '20':
        cl = 'tab:blue'
        label_name = '20th'
        dir_name = os.path.join(plot_dir, 'input_layer3.1.conv1_' + '20th.csv')
    elif label == '80':
        cl = 'tab:green'
        label_name = '80th'
        dir_name = os.path.join(plot_dir, 'input_layer3.1.conv1_' + '80th.csv')
    elif label == '99':
        cl = 'tab:red'
        label_name = '99th'
        dir_name = os.path.join(plot_dir, 'input_linear_' + '99th.csv')

    matrix = []
    for i in range(1):
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

    fig = plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # fig, (ax2) = plt.subplots(ncols=1)
    # fig.tight_layout(pad=0, w_pad=0.5)

x = plot_fid_shade(ax1, relu_dir, label1, 0)
x = plot_fid_shade(ax1, relu_dir, label2, 0)
x = plot_fid_shade(ax1, relu_dir, label3, 0)
x = plot_fid_shade(ax1, relu_dir, label4, 0)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Averaged percentile')
ax1.set_title('ReLU')

x = plot_fid_shade(ax2, aespa_dir, label1, 0)
x = plot_fid_shade(ax2, aespa_dir, label2, 0)
x = plot_fid_shade(ax2, aespa_dir, label3, 0)
x = plot_fid_shade(ax2, aespa_dir, label4, 0)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Averaged percentile')
ax2.set_title('Basis-wise')

x = plot_fid_shade(ax3, postact_dir, label1, 0)
x = plot_fid_shade(ax3, postact_dir, label2, 0)
x = plot_fid_shade(ax3, postact_dir, label3, 0)
x = plot_fid_shade(ax3, postact_dir, label4, 0)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Averaged percentile')
ax3.set_title('PostAct')

# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='right')  # , ncol=4)
fig.set_size_inches(20, 5)


# fig.tight_layout()
fig.savefig('{}/percentile_cos_{}.pdf'.format(
    image_folder, plot_type))
