import random
import numpy as np
import torch.nn
from tqdm import tqdm
from net.dyh import DyHModel
from utils.data_loader import DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from utils.loss_option import loss_node_calculator, loss_edge_calculator, loss_time_calculator, sample_dot


class Option(object):
    def __init__(self):
        self.log_file = "log.txt"
        self.batch_size = 1024
        self.train_file = "dataset/math/math_edge_train.txt"
        self.test_file = "dataset/math/math_edge_val_lr_train_test.txt"
        self.device = "cuda"
        self.g_nums = 11
        self.dim = 32
        self.sub_dim = 32
        self.head = 4
        self.max_epoch = 100
        self.lr = 1e-4
        self.stop_epoch = 5
        self.lam = [1e-8, 1e-8, 1e-8]
        self.tao = [0.1, 0.1, 0.1]

    def __str__(self):
        return str(self.__dict__)


def get_y(loader, edges, neg_samples):
    f = [((src, dst), 1) for src, dst in edges] + [((src, dst), 0) for src, dst in neg_samples]
    random.shuffle(f)
    y = torch.FloatTensor([m[1] for m in f])
    s = [m[0] for m in f]
    if loader.device == "cuda":
        y = y.cuda()
    return [m[1] for m in f], y, s


def main():
    opt = Option()
    print(str(opt))
    with open(opt.log_file, "a") as f:
        f.write(str(opt) + "\n")
    loader = DataLoader(opt)
    model = DyHModel(loader.node_num, opt.dim, opt.dim, opt.head, loader.type_num, opt.dim, opt.sub_dim, 1, opt.dim,
                     opt.dim, opt.dim, opt.dim, opt.sub_dim, 1)
    loss_func = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.device == "cuda":
        model = model.cuda()
        loss_func = loss_func.cuda()
    best_flag = 0.0
    flag = 0
    best_ap = 0.0
    best_acc = 0.0
    best_auc = 0.0
    for epoch_i in range(opt.max_epoch):
        ls, acc = train_epoch(loader, opt, model, loss_func, optim)
        print(f"[{epoch_i}] loss: {ls}, acc: {acc}")
        with open(opt.log_file, "a") as f:
            f.write(f"[{epoch_i}] loss: {ls}, acc: {acc}\n")
        with torch.no_grad():
            valid_ap, valid_auc, valid_acc = valid_epoch(loader, opt, model)
            test_ap, test_auc, test_acc = test_epoch(loader, opt, model)
            print(
                f"[{epoch_i}] valid_acc: {valid_acc}, valid_ap: {valid_ap}, valid_auc: {valid_auc}, test_acc: {test_acc}, test_ap: {test_ap}, test_auc: {test_auc}")
            with open(opt.log_file, "a") as f:
                f.write(
                    f"[{epoch_i}] valid_acc: {valid_acc}, valid_ap: {valid_ap}, valid_auc: {valid_auc}, test_acc: {test_acc}, test_ap: {test_ap}, test_auc: {test_auc}\n")
        if best_flag < valid_ap:
            best_flag = valid_ap
            best_ap = test_ap
            best_auc = test_auc
            best_acc = test_acc
            flag = 0
        if flag == opt.stop_epoch:
            break
        flag += 1
    print(f"best_acc: {best_acc}, best_ap: {best_ap}, best_auc: {best_auc}")
    with open(opt.log_file, "a") as f:
        f.write(f"\nbest_acc: {best_acc}, best_ap: {best_ap}, best_auc: {best_auc}\n")


# def loss_calculate(emb, node_feature_list, edge_feature_list, gru_f, lstm_f, graphs, lam, tao):
#     loss_node_out, loss_node_in = loss_sample(graphs, node_feature_list, emb, tao[0])
#     loss_edge_out, loss_edge_in = loss_sample(graphs, edge_feature_list, emb, tao[1])
#     loss_gru, loss_lstm = loss_double_team(graphs, gru_f, lstm_f, tao[2])
#     return lam[0](loss_node_out - loss_node_in) + lam[1](loss_edge_out - loss_edge_in) + lam[2](loss_gru + loss_lstm)


def train_epoch(loader, opt, model, loss_func, optim):
    model.train()
    neg_samples = loader.train_edges_n
    labels, y, s = get_y(loader, loader.train_edges, neg_samples)
    batch_num = (len(s) - 1) // opt.batch_size + 1
    ls = 0.0
    count = 0.0
    n = len(s)
    for i in tqdm(range(batch_num)):
        node_gat_list, node_gcn_list, edge_att_list, edge_gcn_list, snap_features, gru_f, lstm_f, outputs = model(
            loader.train_snapshots)
        b_n = len(s[i * opt.batch_size: i * opt.batch_size + opt.batch_size])
        dots = sample_dot(s[i * opt.batch_size: i * opt.batch_size + opt.batch_size], outputs)
        loss_node = loss_node_calculator(loader.train_snapshots_dict, node_gat_list, node_gcn_list, opt.tao[0])
        loss_edge = loss_edge_calculator(loader.train_snapshots_dict_iso, edge_att_list, edge_gcn_list, opt.tao[1])
        loss_time = loss_time_calculator(loader.train_snapshots_dict_iso, gru_f, lstm_f, opt.tao[2])
        loss_main = loss_func(dots, y[i * opt.batch_size: i * opt.batch_size + opt.batch_size])
        loss = loss_main + opt.lam[0] * loss_node + opt.lam[1] * loss_edge + opt.lam[2] * loss_time
        y_pred = dots > 0.5
        b_count = (y_pred == y[i * opt.batch_size: i * opt.batch_size + opt.batch_size]).sum().item()
        count += b_count
        loss.backward()
        optim.step()
        ls += loss.detach().cpu().item() * b_n
    return ls / n, count / n


def valid_epoch(loader, opt, model):
    model.eval()
    neg_samples = loader.valid_edges_n
    labels, y, s = get_y(loader, loader.valid_edges, neg_samples)
    batch_num = (len(s) - 1) // opt.batch_size + 1
    pred = list()
    for i in tqdm(range(batch_num)):
        node_gat_list, node_gcn_list, edge_att_list, edge_gcn_list, snap_features, gru_f, lstm_f, outputs = model(
            loader.train_snapshots)
        dots = sample_dot(s[i * opt.batch_size: i * opt.batch_size + opt.batch_size], outputs)
        pred.extend(dots.detach().cpu().tolist())
    ap, auc, acc = evaluate_score(labels, pred)
    return ap, auc, acc


def test_epoch(loader, opt, model):
    model.eval()
    neg_samples = loader.test_edges_n
    labels, y, s = get_y(loader, loader.test_edges, neg_samples)
    batch_num = (len(s) - 1) // opt.batch_size + 1
    pred = list()
    for i in tqdm(range(batch_num)):
        node_gat_list, node_gcn_list, edge_att_list, edge_gcn_list, snap_features, gru_f, lstm_f, outputs = model(
            loader.train_snapshots)
        dots = sample_dot(s[i * opt.batch_size: i * opt.batch_size + opt.batch_size], outputs)
        pred.extend(dots.detach().cpu().tolist())
    ap, auc, acc = evaluate_score(labels, pred)
    return ap, auc, acc


def evaluate_score(labels, pred):
    labels_np = np.array(labels)
    labels_pred = np.array(pred)
    auc = roc_auc_score(labels_np, labels_pred)
    ap = average_precision_score(labels_np, labels_pred)
    acc = ((labels_pred > 0.5) == labels_np).sum().tolist() / len(labels)
    return ap, auc, acc


if __name__ == '__main__':
    main()
