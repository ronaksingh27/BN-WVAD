import pdb
import numpy as np
import torch.utils.data as data
import utils
import time
import wandb

from options import *
from config import *

from train import train
from losses import LossComputer
from test import test
from models import WSAD

from dataset_loader import *
from tqdm import tqdm

localtime = time.localtime()
time_ymd = time.strftime("%Y-%m-%d", localtime)
time_hms = time.strftime("%H:%M:%S", localtime)

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    
    config = Config(args)
    worker_init_fn = None
    gpus = [0]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    config.len_feature = 1024
    net = WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    net = net.cuda()

    normal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {'step': [], 'AUC': [], 'AP': []}
    
    best_auc = 0

    criterion = LossComputer()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr[0],
        betas = (0.9, 0.999), weight_decay = args.weight_decay)

    best_scores = {
        'best_AUC': -1,
        'best_AP': -1,
    }

    metric = test(net, test_loader, test_info, 0)
    for step in tqdm(
            range(1, args.num_iters + 1),
            total = args.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and args.lr[step - 1] != args.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        losses = train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion)
        wandb.log(losses, step=step)
        if step % args.plot_freq == 0 and step > 0:
            metric = test(net, test_loader, test_info, step)

            if test_info["AP"][-1] > best_scores['best_AP']:
                utils.save_best_record(test_info, os.path.join(args.log_path, "xd_best_record_{}.txt".format(args.seed)))

                torch.save(net.state_dict(), os.path.join(args.model_path, "xd_best_{}.pkl".format(args.seed)))
            
            for n, v in metric.items():
                best_name = 'best_' + n
                best_scores[best_name] = v if v > best_scores[best_name] else best_scores[best_name]
