from __future__ import division

import argparse
import pickle as pk
import time
from operator import itemgetter

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from torch.utils.data import DataLoader

from util import print_metric, chunks, print_network, read_ic, plot_metric
from model import CNFNet


def load_args():
    """load training parameters"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('--thres', type=int, default=200, help='censor threshold')
    parser.add_argument('--alpha', type=float, default=0, help='')
    parser.add_argument('--beta', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=float, default=12, help='')
    parser.add_argument('--num_feat', type=float, default=16, help='')
    parser.add_argument('--energy_input_dim', type=float, default=17, help='')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=8, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--data', type=str, default='c432', help='Dataset name')
    parser.add_argument('--censor_ratio', type=float, default=0.1, help='censor ratio')

    args = parser.parse_args()

    print(args)
    return args


def padding_and_trim_reg(model, l, inc_feat, feats, batch_size):
    last_entry_size = l[-1].shape[0]

    if last_entry_size != batch_size:
        l[-1] = torch.cat((l[-1], torch.LongTensor([l[-1][0]]).repeat(batch_size - l[-1].shape[0])))
        output = [model.get_reg(itemgetter(*_)(inc_feat), feats[_]) for _ in l]
        output = torch.cat(output).data.numpy().flatten()[:-(batch_size - last_entry_size)]
        l[-1] = l[-1][:last_entry_size]
    else:
        output = [model.get_reg(itemgetter(*_)(inc_feat), feats[_]) for _ in l]
        output = torch.cat(output).data.numpy().flatten()

    return output


def padding_and_trim_class(model, l, inc_feat, feats, batch_size):
    last_entry_size = l[-1].shape[0]

    if last_entry_size != batch_size:
        l[-1] = torch.cat((l[-1], torch.LongTensor([l[-1][0]]).repeat(batch_size - l[-1].shape[0])))
        output = [model.get_class(itemgetter(*_)(inc_feat), feats[_]) for _ in l]
        output = torch.cat(output).data.numpy().flatten()[:-(batch_size - last_entry_size)]
        l[-1] = l[-1][:last_entry_size]
    else:
        output = [model.get_class(itemgetter(*_)(inc_feat), feats[_]) for _ in l]
        output = torch.cat(output).data.numpy().flatten()

    return output


def main(args):
    # select benchmark
    c = args.data

    # load data
    inc_feat, feats, times, censored_train_num, censored_test_num, uncensored_train_num, uncensored_test_num = read_ic(
        args)

    # initialize model parameter
    model = CNFNet(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print_network(model)

    uncensor_loader = DataLoader(uncensored_train_num, batch_size=args.batch_size, shuffle=True, drop_last=True)
    censor_loader = DataLoader(censored_train_num, batch_size=args.batch_size, shuffle=True, drop_last=True)

    args.num_instance = len(times)

    def train(model):
        for epoch in range(args.epochs):
            for step, un_ids in enumerate(uncensor_loader):
                t = time.time()

                c_ids = next(iter(censor_loader))

                model.train()
                output = model(itemgetter(*un_ids)(inc_feat), itemgetter(*c_ids)(inc_feat), feats[un_ids], feats[c_ids])
                loss_train, ret_str = model.combo_loss(output, times[un_ids].view(-1, 1))

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                print('Epoch: {:02d}/{:04d}'.format(epoch, step + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'time: {:.4f}s'.format(time.time() - t), ret_str)

    ############
    # Train model
    ############
    t_total = time.time()

    train(model)

    # print training info
    # plot_metric(range(len(train_loss)), train_loss, eval_loss,
    #             '{}_{}_{}_train'.format(c, args.sa_loss, args.censor_ratio),
    #             '{}_{}_{}_eval'.format(c, args.sa_loss, args.censor_ratio))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def test(censored_test_num, uncensored_test_num):
        model.eval()

        test_un_ids = list(chunks(uncensored_test_num, args.batch_size))
        test_c_ids = list(chunks(censored_test_num, args.batch_size))

        output_un = padding_and_trim_reg(model, test_un_ids, inc_feat, feats, args.batch_size)
        output_c_pos = padding_and_trim_class(model, test_c_ids, inc_feat, feats, args.batch_size)
        output_c_neg = padding_and_trim_class(model, test_un_ids, inc_feat, feats, args.batch_size)

        prefix = "{}_{}_{}_{}_{}".format('DSAG', c, args.thres, args.alpha, args.beta)
        # uncensor test
        time_un_test = times[torch.cat(test_un_ids)].data.numpy()
        time_un_pred = output_un
        # print(time_un_test, time_un_pred}
        print("test loss: {} ".format(mean_squared_error(time_un_pred, time_un_test)))
        plot_metric(range(time_un_test.size), time_un_pred, time_un_test,
                    '{}_pred'.format(prefix),
                    '{}_real'.format(prefix))

        pk.dump(print_metric(time_un_test, time_un_pred),
                open('{}_metric.pk'.format(prefix), 'wb'))
        pk.dump(time_un_test, open('{}_test_list.pk'.format(prefix), 'wb'))
        pk.dump(time_un_pred, open('{}_pred_list.pk'.format(prefix), 'wb'))

        # censor test
        censor_size, uncensor_size = censored_test_num.shape[0], uncensored_test_num.shape[0]
        pred = np.concatenate([output_c_pos, output_c_neg])
        censor_labels = torch.cat(
            (torch.FloatTensor([0]).repeat(uncensor_size), torch.FloatTensor([1]).repeat(censor_size)))
        confuse_mat = confusion_matrix(pred, censor_labels)
        class_report = classification_report(pred, censor_labels)
        pk.dump(confuse_mat, open('{}_confusion_matrix.pk'.format(prefix), 'wb'))
        pk.dump(class_report, open('{}_class_report.pk'.format(prefix), 'wb'))

    ############
    # Testing
    ############
    test(censored_test_num, uncensored_test_num)


if __name__ == '__main__':
    args = load_args()

    print("\n====================\n{}".format(args.data))
    main(args)
