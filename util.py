import pickle as pk

import torch
from pylab import *
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from torch.utils.data import Dataset


def read_ic(args):
    """read preprocessed data files"""

    root_dir = './preprocess_data/{}_{}.pk'
    c = args.data
    times = np.array(pk.load(open(root_dir.format(c, 'Y'), 'rb')))
    input = np.array(pk.load(open(root_dir.format(c, 'X'), 'rb')))

    inc_feat = [_[-7:] for _ in input]
    feat = [_[:-7] for _ in input]

    print('Data size: {}'.format(times.size))

    DATA_NUM = len(times)
    train_rate = 0.8
    test_rate = 1

    data_ind = np.arange(DATA_NUM)
    np.random.seed(8)

    # censored_train_num = sorted(data_ind[range(int(DATA_NUM * train_rate * args.censor_ratio))])
    # train_num = sorted(data_ind[range(
    #     int(DATA_NUM * train_rate))])
    # test_num = sorted(data_ind[range(
    #     int(DATA_NUM * train_rate),
    #     int(DATA_NUM * test_rate))])

    # OPT 1: censor by threshold
    threshold = args.thres
    data_max, data_min = np.max(times), np.min(times)

    censored_num = sorted(np.where(times >= threshold)[0])
    np.random.shuffle(censored_num)
    censored_num_size = len(censored_num)

    censored_train_num = sorted(censored_num[:int(censored_num_size * train_rate)])
    censored_test_num = sorted(censored_num[int(censored_num_size * train_rate):])

    uncensored_num = sorted(np.where(times < threshold)[0])
    np.random.shuffle(uncensored_num)
    uncensored_num_size = len(uncensored_num)

    uncensored_train_num = sorted(uncensored_num[:int(uncensored_num_size * train_rate)])
    uncensored_test_num = sorted(uncensored_num[int(uncensored_num_size * train_rate):])

    print("Threshold: {}, Censored/Uncensored: {}/{}, Max/Min: {}/{}".format(threshold, censored_num_size,
                                                                             uncensored_num_size, data_max, data_min))

    censored_train_num = torch.LongTensor(censored_train_num)
    censored_test_num = torch.LongTensor(censored_test_num)
    uncensored_train_num = torch.LongTensor(uncensored_train_num)
    uncensored_test_num = torch.LongTensor(uncensored_test_num)
    times = torch.FloatTensor(np.log1p(times))
    feat = torch.FloatTensor(feat)

    return inc_feat, feat, times, censored_train_num, censored_test_num, uncensored_train_num, uncensored_test_num


def print_network(net):
    """print brief structure of neural networks"""

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def print_metric(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    evs = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pc = pearsonr(y_test, y_pred)
    sc = spearmanr(y_test, y_pred)

    print(
        "MSE/RMSE: {:.4f}/{:.4f}, MAE: {:.4f}, EVS: {:.4f}, R2: {:.4f}, Corr: {:.4f}/{:.4f}".format(mse, rmse, mae,
                                                                                                    evs, r2, pc[0],
                                                                                                    sc[0]))

    return mse, rmse, evs, mae, r2, pc[0], sc[0]


def plot_metric(x, y, z, yc='train', zc='eval'):
    """plot result image and save in local files"""

    matplotlib.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plt.scatter(x, y)
    # plot(x, y, markerfacecolor='salmon', markeredgewidth=1, markevery=slice(40, len(y), 70),
    #      linestyle=':', marker='o', color='crimson', linewidth=3, label='fit')  # fit result
    plot(x, y, color='crimson', linewidth=5, label=yc)  # fit result

    # plt.scatter(x, z)
    # plot(x, y, markerfacecolor='salmon', markeredgewidth=1, markevery=slice(40, len(y), 70),
    #      linestyle=':', marker='o', color='crimson', linewidth=3, label='fit')  # fit result
    plot(x, z, color='blue', linewidth=5, label=zc)  # fit result
    plt.legend(loc="best", prop={'size': 20})
    # plt.savefig('loss.png')
    plt.savefig('{}-{}.png'.format(yc, zc))
    plt.close()


class GraphDataset(Dataset):
    """wrap function for sampling training instances"""

    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, index):
        return self.ids[index]

    def __len__(self):
        return len(self.ids)


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def store_model(model, name='test'):
    torch.save(model, '{}'.format(name))


def restore_model(path):
    model = torch.load(path)
    model.eval()

    return model
