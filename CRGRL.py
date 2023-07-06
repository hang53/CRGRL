import copy
import torch
import argparse
from datasets import SPMotif
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, BatchNorm, fps
from utils.mask import set_masks, clear_masks
from train import Graph_draw

from torch_scatter import scatter
from torch.nn import Linear, ReLU, ModuleList, Softmax, CrossEntropyLoss

import os
import numpy as np
import os.path as osp
from torch.autograd import grad
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.get_subgraph import drop_info_return_full, relabel
from utils.calculaters import calculate_percentages, find_matching_elements
from gnn import SPMotifNet
from gnn import SupConLoss
import time


class QNet(nn.Module):  # The q(.) for calculating IVs

    def __init__(self, ):
        super(QNet, self).__init__()
        self.convq1 = LEConv(in_channels=4, out_channels=args.channels)
        self.convq2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp_edge = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 4)
        )

        self.mlp_node = nn.Sequential(
            nn.Linear(args.channels, args.channels * 2),
            nn.ReLU(),
            nn.Linear(args.channels * 2, 1)
        )

    def forward(self, data, en_node_weight=False, ):

        q = self.convq1(data.x, data.edge_index, data.edge_attr.view(-1))
        q = self.convq2(q, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([q[row], q[col]], dim=-1)
        edge_weight = self.mlp_edge(edge_rep)  # The edge wights as IVs

        node_weight = self.mlp_node(q).view(-1)

        if en_node_weight:
            return edge_weight, node_weight
        else:
            return edge_weight


def zero_elements_based_on_percent(tensor1, tensor2, percent):
    assert tensor1.size(0) == tensor2.size(0), "第一维度大小不一致"

    # 获取要置零的元素数量
    num_elements = tensor1.size(0)
    num_zeros = int(num_elements * percent)

    # 根据排序后的索引将后一定比例的元素置为0
    _, indices = torch.sort(tensor1)
    tensor2[indices[-num_zeros:], ...] = 0

    return tensor2

def normalize_tensor(X):
    # 计算最小值和范围
    min_val = torch.min(X)
    range_val = torch.max(X) - min_val

    # 规范化张量到0.7和1之间
    normalized_X = 0.7 + ((X - min_val) / range_val) * 0.3

    return normalized_X


def zero_last_percent(tensor, percent):
    assert 0 <= percent <= 1, "百分比应在0到1之间"

    # 计算需要置为0的元素数量
    num_elements = int(tensor.size(0) * percent)

    # 按大小排列张量的元素
    sorted_indices = torch.argsort(tensor)

    # 创建一个布尔张量，大小排在后 num_elements 的元素为 False，其余元素为 True
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[sorted_indices[-num_elements:]] = False

    # 使用 torch.where() 实现原地操作，将大小排在后 num_elements 的元素置为0
    tensor = torch.where(mask, tensor, torch.zeros_like(tensor))

    return tensor

def project_to_01(x):
    sigmoid = torch.nn.Sigmoid()
    projected = sigmoid(x)
    return projected


def remove_x_elements(edge_score, edge_index, x):
    assert edge_score.size(0) == edge_index.size(1), "edge_score和edge_index的第一维度不相同"

    # 将输入参数移到相同的设备上
    device = edge_score.device
    # edge_index = edge_index.to(device)
    # x = x.to(device)

    # 获取值不为0的元素的索引
    nonzero_indices = torch.nonzero(edge_score)

    # 获取索引对应的元素
    selected_edge_index = edge_index[:, nonzero_indices.squeeze()]

    # 去除重复的元素，得到索引
    unique_indices = torch.unique(selected_edge_index)

    # 将不在unique_indices中的x值置为0
    indices = torch.arange(x.size(0), device=device)
    mask = torch.logical_not(torch.any(indices.unsqueeze(1) == unique_indices.unsqueeze(0), dim=1))
    x[mask] = 0

    return x


class ProcessNet(nn.Module):

    def __init__(self, drop):
        super(ProcessNet, self).__init__()
        self.conv1 = LEConv(in_channels=4, out_channels=args.channels)
        self.conv2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1)
        )
        self.d = drop

    def forward(self, data, edge_score):
        # batch_norm
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        (robust_edge_index, robust_edge_attr, robust_edge_weight), \
        (full_edge_index, full_edge_attr, full_edge_weight) = drop_info_return_full(data, edge_score, self.d)  # r

        robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index, data.batch)
        full_x, full_edge_index, full_batch, _ = relabel(x, full_edge_index, data.batch)

        return (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
               (full_x, full_edge_index, full_edge_attr, full_edge_weight, full_batch), \
               edge_score


class BasicNet(nn.Module):
    def __init__(self, drop):
        super(BasicNet, self).__init__()
        self.conv1 = LEConv(in_channels=4, out_channels=args.channels)
        self.conv2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.conv3 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.conv4 = LEConv(in_channels=args.channels, out_channels=args.channels)

        self.mlp = torch.nn.Sequential(
            Linear(args.channels, 2 * args.channels),
            ReLU(),
            Linear(2 * args.channels, num_classes)
        )

        self.d = drop

    def forward(self, data, edge_score, en_full = False, dis_drop = False):
        # dis_drop = False
        if en_full:
            # conv layer1
            x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))

            x = F.relu(self.conv2(x, data.edge_index, data.edge_attr.view(-1)))

            x = F.relu(self.conv3(x, data.edge_index, data.edge_attr.view(-1)))

            x = F.relu(self.conv4(x, data.edge_index, data.edge_attr.view(-1)))

            assert not torch.any(torch.isnan(x))
            # clear_masks(self.conv4)

            # pooling layer
            size = int(data.batch.max().item() + 1)
            grah_feature = scatter(x, data.batch, dim=0, dim_size=size, reduce='mean')

            # prediction
            pred = self.mlp(grah_feature)
        elif dis_drop:

            # calculate drop percent
            # percent = calculate_percentages(edge_score, (self.d / 4) )

            # conv layer1
            edge_score_ = edge_score[:, 0]


            x = F.relu(self.conv1(data.x, data.edge_index, project_to_01(edge_score_)))

            edge_score_0 = edge_score[:, 1]

            x = F.relu(self.conv2(x, data.edge_index, project_to_01(edge_score_0)))

            edge_score_1 = edge_score[:, 2]

            x = F.relu(self.conv3(x, data.edge_index, project_to_01(edge_score_1)))

            edge_score_2 = edge_score[:, 3]

            x = F.relu(self.conv4(x, data.edge_index, project_to_01(edge_score_2)))

            # robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index_1, data.batch)

            assert not torch.any(torch.isnan(x))
            # clear_masks(self.conv4)

            # pooling layer
            size = int(data.batch.max().item() + 1)
            grah_feature = scatter(x, data.batch, dim=0, dim_size=size, reduce='mean')

            # prediction
            pred = self.mlp(grah_feature)

        else:
            # calculate drop percent
            # percent = calculate_percentages(edge_score, (self.d / 4) )

            # conv layer1
            edge_score_ = edge_score[:, 0]
            x = F.relu(self.conv1(data.x, data.edge_index, project_to_01(edge_score_)))

            edge_score_0 = edge_score[:, 1]
            # edge_score_0 = F.normalize(edge_score_0, p=2, dim=0)
            # (robust_edge_index_0, robust_edge_attr_0, robust_edge_weight_0), \
            # (full_edge_index, full_edge_attr, full_edge_weight) = \
            #     drop_info_return_full(data, edge_score_0, 0.1) # r
            # robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index, data.batch)


            x = F.relu(self.conv2(x, data.edge_index, project_to_01(edge_score_0)))
            # print(edge_score_0)

            edge_score_1 = edge_score[:, 2]
            # edge_score_1 = F.normalize(edge_score_1, p=2, dim=0)
            (robust_edge_index_1, robust_edge_attr_1, robust_edge_weight_1, reserve_index_1), \
            (full_edge_index, full_edge_attr, full_edge_weight) = \
                drop_info_return_full(data, edge_score_1, 0.75, require_edge_reserve_index=True) # r1

            # robust_edge_index_1, robust_edge_weight_1 = \
                # find_matching_elements(robust_edge_index_0, robust_edge_index_1, robust_edge_weight_1)

            x = F.relu(self.conv3(x, robust_edge_index_1, project_to_01(robust_edge_weight_1)))

            edge_score_2 = edge_score[:, 3]
            # edge_score_2 = F.normalize(edge_score_2, p=2, dim=0)

            # (robust_edge_index_2, robust_edge_attr_2, robust_edge_weight_2), \
            # (full_edge_index, full_edge_attr, full_edge_weight) = \
            #     drop_info_return_full(data, edge_score_2, 0.1) # r
            # # robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index, data.batch)
            # robust_edge_index_2, robust_edge_weight_2 = \
            #     find_matching_elements(robust_edge_index_1, robust_edge_index_2, robust_edge_weight_2)
            # robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index_2, data.batch)

            x = F.relu(self.conv4(x, robust_edge_index_1, project_to_01(edge_score_2[reserve_index_1]) ))

            robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index_1, data.batch)

            assert not torch.any(torch.isnan(x))
            # clear_masks(self.conv4)

            # pooling layer
            size = int(data.batch.max().item() + 1)
            grah_feature = scatter(robust_x, robust_batch, dim=0, dim_size=size, reduce='mean')

            # prediction
            pred = self.mlp(grah_feature)

        return grah_feature, pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training RCGRL')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='/home/gaohang/Researches/CaCL/RCGRL/data/', type=str,
                        help='directory for datasets.')  # dataset location
    parser.add_argument('--epoch', default=300, type=int, help='training iterations')
    parser.add_argument('--seed', nargs=3, default=5, help='random seed')
    parser.add_argument('--channels', default=32, type=int, help='width of network')
    parser.add_argument('--bias', default='0.7', type=str, help='select bias extend')
    parser.add_argument('--pretrain', default=30, type=int, help='pretrain epoch')
    parser.add_argument('--lambda_Lc', default=1, type=float, help='lambda of Lc')
    parser.add_argument('--gamma_Lr', default=0.1, type=float, help='gamma of Lr')
    parser.add_argument('--tau_Lr', default=1000, type=float, help='gamma of Lr')
    parser.add_argument('--drop', default=0.75, type=float, help='percentage of data droped')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate for the predictor')
    args = parser.parse_args()
    # dataset
    num_classes = 3
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    train_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    n_train_data, n_val_data = len(train_dataset), len(val_dataset)
    n_test_data = float(len(test_dataset))

    best_val = 0
    best_val_test = 0

    set_seed(args.seed)

    result_list = []

    for i in range(1):
        best_val = 0
        set_seed(i)
        f_after = SPMotifNet(args.channels).to(
            device)  # divided f(.) into two parts, before confounder removal and after
        q_net = QNet().to(device)
        f_before = ProcessNet(args.drop).to(device)

        f_basic = BasicNet(args.drop).to(device)

        model_optimizer = torch.optim.Adam(
            list(f_after.parameters()) +
            list(q_net.parameters()) +
            list(f_basic.parameters()) +
            list(f_before.parameters()),
            lr=args.net_lr)
        CELoss = nn.CrossEntropyLoss(reduction="mean")
        EleCELoss = nn.CrossEntropyLoss(reduction="none")
        MSELoss = nn.MSELoss(reduction="none")
        SupConLoss_Func = SupConLoss()
        Cosine_Sim = nn.CosineSimilarity()


        def train_mode():
            f_after.train()
            f_before.train()
            q_net.eval()
            f_basic.train()


        def train_q_mode():
            f_after.eval()
            f_before.eval()
            q_net.train()
            f_basic.eval()


        def val_mode():
            f_after.eval()
            f_before.eval()
            q_net.eval()
            f_basic.eval()


        def test_acc(loader, q_net, predictor):
            acc = 0
            for graph in loader:
                graph.to(device)

                edge_score, node_score = q_net(graph, en_node_weight=True)

                robust_rep, robust_out = f_basic(graph, edge_score)

                acc += torch.sum(robust_out.argmax(-1).view(-1) == graph.y.view(-1))
            acc = float(acc) / len(loader.dataset)
            return acc


        print(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")
        cnt, last_val_acc = 0, 0

        for epoch in range(args.epoch):

            # udpate f
            print("[", time.localtime(),"]")

            print("\n# update f:")

            all_loss, n_bw = 0, 0
            all_batched_loss, all_var_loss = 0, 0
            train_mode()
            x = 0
            counter = 0

            saved = 0

            for graph in train_loader:

                n_bw += 1
                graph.to(device)
                N = graph.num_graphs

                edge_score, node_score = q_net(graph, en_node_weight=True)

                if saved == 0:
                    torch.save(edge_score, "./draw/save_W_0.7/W_train_epoch_" + str(epoch) + ".pth")
                    saved = 1

                robust_rep, robust_out = f_basic(graph, edge_score)

                full_rep, _ = f_basic(graph, edge_score, en_full = True)

                robust_contrast_feature = robust_rep.unsqueeze(1).clone()
                full_contrast_feature = full_rep.unsqueeze(1).clone()
                contrast_feature = torch.cat((robust_contrast_feature, full_contrast_feature), 1)

                contrast_loss = (-1) * Cosine_Sim(robust_rep.detach(), full_rep)

                robust_loss = CELoss(robust_out, graph.y)
                loss_r = robust_loss

                all_batched_loss += loss_r

            # print( x / counter )

            all_batched_loss /= n_bw
            all_loss = all_batched_loss
            model_optimizer.zero_grad()
            print("all loss is:", all_loss)
            all_loss.backward()
            model_optimizer.step()

            # update q

            print("[", time.localtime(), "]")

            print("# update q:")

            all_loss, n_bw = 0, 0
            all_batched_loss, all_var_loss = 0, 0
            train_q_mode()
            for graph in train_loader:
                n_bw += 1
                graph.to(device)
                N = graph.num_graphs
                edge_score = q_net(graph)

                n_bw += 1
                graph.to(device)
                N = graph.num_graphs

                edge_score, node_score = q_net(graph, en_node_weight=True)
                robust_rep, robust_out = f_basic(graph, edge_score, dis_drop=True)

                # full_rep, _ = f_basic(graph, edge_score, en_full=True, dis_drop=True)

                robust_loss = CELoss(robust_out, graph.y)
                robust_regular_mean = EleCELoss(robust_out, graph.y)

                robust_regular_target = robust_loss.expand_as(robust_regular_mean)
                robust_regular_mse = MSELoss(robust_regular_mean, robust_regular_target)

                robust_regular_mse_grouped_target = robust_regular_mean
                for i in range(num_classes):
                    index = torch.nonzero(graph.y == float(i)).squeeze()
                    robust_regular_mse_grouped_target[index] = torch.mean(robust_regular_mean[index])
                robust_regular = EleCELoss(robust_out, graph.y)

                robust_sp_feature = robust_rep.unsqueeze(1)
                robust_sp_feature = torch.cat((robust_sp_feature, robust_sp_feature), 1)
                robust_sp_loss = SupConLoss_Func(robust_sp_feature, graph.y)

                robust_weight_loss_target = robust_rep.clone()
                for i in range(num_classes):
                    index = torch.nonzero(graph.y == float(i)).squeeze()
                    robust_weight_loss_target[index] = torch.mean(robust_rep[index].clone(), dim=0)
                robust_weight = Cosine_Sim(robust_rep, robust_weight_loss_target)

                robust_sim_all_target = robust_rep.clone()
                robust_sim_all_target[:] = torch.mean(robust_rep.clone(), dim=0)
                robust_sim_all = Cosine_Sim(robust_rep, robust_sim_all_target)
                robust_sim_all = (1 - robust_sim_all) / 2
                robust_sim_all = robust_sim_all / robust_sim_all.shape[0]
                robust_sim_all = robust_sim_all.sum()

                robust_variance = torch.var(robust_rep, dim=1)
                robust_variance = robust_variance.sum().div(robust_rep.shape[0])

                if (robust_sim_all+robust_variance) < 0.01:
                    sim_all_e = -1 * args.tau_Lr
                else:
                    sim_all_e = 0

                correct_index = torch.nonzero(robust_out.argmax(-1).view(-1) == graph.y.view(-1)).squeeze()
                wrong_index = torch.nonzero(robust_out.argmax(-1).view(-1) != graph.y.view(-1)).squeeze()
                robust_weight[correct_index] = 1 - robust_weight[correct_index]
                robust_weight[wrong_index] = robust_weight[wrong_index] + 1
                robust_weight = robust_weight / 2
                robust_weight = robust_weight.pow(args.gamma_Lr).detach()
                robust_regular_weighted = robust_regular * robust_weight

                loss_c = robust_regular_weighted.sum() + sim_all_e * (robust_sim_all+robust_variance)  # Lc

                all_batched_loss += loss_c

            all_batched_loss /= n_bw
            all_loss = all_batched_loss
            print("loss is:", all_loss)
            model_optimizer.zero_grad()
            all_loss.backward()
            model_optimizer.step()

            print("[", time.localtime(), "]")

            print("# test:")
            val_mode()
            robust_rep_all = []
            label_all = []
            with torch.no_grad():

                # train_acc = test_acc(train_loader, q_net, f_after)
                train_acc = 0
                val_acc = test_acc(val_loader, q_net, f_after)
                robust_acc = 0.
                saved = 0
                for graph in test_loader:
                    graph.to(device)
                    edge_score, node_score = q_net(graph, en_node_weight=True)

                    if saved ==0:
                        torch.save(edge_score, "./draw/save_W_0.7/W_test_epoch_" + str(epoch) + ".pth")
                        saved = 1

                    robust_rep, robust_out = f_basic(graph, edge_score)
                    robust_rep_all.append(robust_rep)
                    label_all.append(graph.y)

                    robust_acc += torch.sum(robust_out.argmax(-1).view(-1) == graph.y.view(-1)) / n_test_data

                print("Epoch {:3d}  all_loss:{:2.3f}  "
                      "Train_ACC:{:.3f} Test_ACC{:.3f}  Val_ACC:{:.3f}  ".format(
                    epoch, all_loss,
                    train_acc, robust_acc, val_acc
                ))

            # print("val_acc:", val_acc, "last_val_acc:", last_val_acc, "cnt:", cnt)

            print("[", time.localtime(),"]")
            robust_rep_all = torch.cat(robust_rep_all)
            label_all = torch.cat(label_all)
            # torch.save(robust_rep_all, "./draw/save_r/robust_rep_all_epoch_"+str(epoch)+".pth")
            # torch.save(label_all, "./draw/save_r/label_all_epoch_" + str(epoch) + ".pth")

            if epoch >= args.pretrain:
                if val_acc < last_val_acc:
                    cnt += 1
                else:
                    cnt = 0
                    last_val_acc = val_acc
            if cnt >= 5:
                print("Early Stop!")
                break
            if val_acc > best_val:
                # print("val_acc > best_val", val_acc, ">", best_val)
                best_val = val_acc
                best_val_test = robust_acc

            print("best val:{:2.3f}, best test: {:2.3f}".format(best_val, best_val_test))

        print("#=================== RESULT: ==============================#")
        print("best val is", best_val)
        print("best test is", best_val_test)
        result_list.append(best_val_test)
        print(result_list)
        print("#==========================================================#")

    print("#===================== ALL RESULT ======================#")
    print(result_list)
    print("#=======================================================#")
