import argparse
import glob
import os
import sys
import logging
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import json
from pathlib import Path
from collections import namedtuple
import time
import uuid
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

from transformers import get_linear_schedule_with_warmup

from torchfm.dataset.pecos_fm_format import PecosFMDataset, pair_collate_fn
from torchfm.model.efm import EmbeddingFactorizationMachineModel

from retrieve import FaissRetriever, search_queries, write_ranking, to_marco
from ms_marco_eval import compute_metrics, compute_metrics_from_files

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("emb_dir", help="Path to embedding data directory (e.g. `XZs/[something]`).", type=str)
    parser.add_argument("lbl_dir", help="Path to label data directory (e.g. `Ys/[something]`).", type=str)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--reg_rate', type=float, default=1e-6, help='Weight decay.')
    parser.add_argument('--nepochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument("--lr_scheduler", help="Learning rate scheduler.", type=str, 
                        default='linear', choices=['linear', None])
    parser.add_argument('--x_emb_dim', type=int, default=768, help='Dimension of input x embedding.')
    parser.add_argument('--z_emb_dim', type=int, default=768, help='Dimension of input x embedding.')
    parser.add_argument('--k', type=int, default=768, help='Number of factors.')
    parser.add_argument("--q_mode", default='asymmetric', type=str, help="Mode for modeling joint multiplication term.")
    parser.add_argument('--identity_factor', default=1, help='Factor for identity when intializing weights.')
    # parser.add_argument('--pos_weight', type=float, default=None, help='Ratio for weighting positive class losses.')
    parser.add_argument('--auto_stop', action='store_true')
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument('--tanh', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--use_fm', action='store_true')
    parser.add_argument('--eval_zeroth_step', action='store_true')
    parser.add_argument('--eval_trn', action='store_true')
    parser.add_argument('--ip', action='store_true')
    parser.add_argument("--Y_trn_suffix", default=None, type=str,
                        help="Suffix for Y train labels, typically for different negatives.")
    parser.add_argument("--Y_tst_suffix", default=None, type=str,
                        help="Suffix for Y test labels, typically for different negatives.")
    
    parser.add_argument('--patience', type=int, default=None, help='Number of steps to wait for improvement of metric till early stopped.')
    parser.add_argument('--inference_bsize', type=int, default=2048, help='Inference batch size.')
    parser.add_argument('--bsize', type=int, default=8, help='Batch size.')
    parser.add_argument('--n_train_zs', type=int, default=8, help='Number of zs per x.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for loading data.')
    parser.add_argument("--save_model_dir", help="directory to save model.", type=str, default=None)
    parser.add_argument("--pretrained_model_dir", help="directory to load pretrained model.", type=str, default=None)
    parser.add_argument("--pretrained_ckpt_path", help="some specific path to load model checkpoint (if None defaults to `pretrained_model_dir/fm.pth`).", type=str, default=None)
    
    parser.add_argument('--eval_interval', type=int, default=20000, help='Log interval.')
    parser.add_argument('--ckpt_interval', type=int, default=250000, help='Checkpoint interval.')
    parser.add_argument('--qrels_trn_tsv', type=str, help='Testing ground truth rank path for evaluating MRR@10 train.', 
                        default='/efs/core-pecos/users/cybai/tevatron/examples/coCondenser-marco/marco/qrels.train.tsv')
    parser.add_argument('--qrels_tst_tsv', type=str, help='Testing ground truth rank path for evaluating MRR@10 dev.', 
                        default='/efs/core-pecos/users/cybai/tevatron/examples/coCondenser-marco/marco/qrels.dev.tsv')
    parser.add_argument("--monitor_metric", default='tst_mrr@10', type=str,
                        help="Test metric to monitor for early stopping.")
    
    return parser.parse_args()

def load_data(emb_dir, lbl_dir, Y_trn_suffix=None, Y_tst_suffix=None, verbose=True):
    
    if os.path.isfile(os.path.join(emb_dir, 'X.trn.npz')) and \
       os.path.isfile(os.path.join(emb_dir, 'X.tst.npz')) and \
       os.path.isfile(os.path.join(emb_dir, 'Z.npz')):
    
        with np.load(os.path.join(emb_dir, 'X.trn.npz')) as f:
            X_trn = {k: v for k, v in f.items()}

        with np.load(os.path.join(emb_dir, 'X.tst.npz')) as f:
            X_tst = {k: v for k, v in f.items()}

        with np.load(os.path.join(emb_dir, 'Z.npz')) as f:
            Z = {k: v for k, v in f.items()}
            
    elif os.path.isdir(os.path.join(emb_dir, 'corpus')) and \
         os.path.isdir(os.path.join(emb_dir, 'query')):
        
        with open(os.path.join(emb_dir, 'query', 'qry.trn.pt'), 'rb') as f:
            embs, indices = pickle.load(f)
        X_trn = {'embs': np.array(embs).astype(np.float32), 'indices': np.array(indices).astype(int)}
        
        with open(os.path.join(emb_dir, 'query', 'qry.dev.pt'), 'rb') as f:
            embs, indices = pickle.load(f)
        X_tst = {'embs': np.array(embs).astype(np.float32), 'indices': np.array(indices).astype(int)}

        # Z
        embs, indices = [], []
        for pkl_path in sorted(glob.glob(os.path.join(emb_dir, 'corpus', 'split*.pt'))):
            with open(pkl_path, 'rb') as f:
                embs_, indices_ = pickle.load(f)
                embs_ = np.array(embs_).astype(np.float32)
                indices_ = np.array(indices_).astype(int)
                
                embs.append(embs_)
                indices.append(indices_)
        Z = {'embs': np.concatenate(embs, axis=0), 'indices': np.concatenate(indices, axis=0)}
        
        del embs, indices
    else:
        raise FileNotFoundError
    
    if Y_trn_suffix is not None:
        Y_trn = sparse.load_npz(os.path.join(lbl_dir, f'Y.trn.{Y_trn_suffix}.npz'))
    else:
        Y_trn = sparse.load_npz(os.path.join(lbl_dir, 'Y.trn.npz'))
        
    if Y_tst_suffix is not None:
        Y_tst = sparse.load_npz(os.path.join(lbl_dir, f'Y.tst.{Y_tst_suffix}.npz'))
    else:
        Y_tst = sparse.load_npz(os.path.join(lbl_dir, 'Y.tst.npz'))
    
    # checking for dimension matching and values.
    assert X_trn['embs'].shape[0] == Y_trn.shape[0]
    assert X_tst['embs'].shape[0] == Y_tst.shape[0]
    assert Z['embs'].shape[0] == Y_trn.shape[1]
    assert Z['embs'].shape[0] == Y_tst.shape[1]
    assert X_trn['embs'].shape[1] == Z['embs'].shape[1]
    assert X_tst['embs'].shape[1] == Z['embs'].shape[1]
    assert set(Y_trn.data) == set([-1, 1])
    assert ((set(Y_tst.data) == set([-1, 1])) or (set(Y_tst.data) == set([1])))
    
    if verbose:
        print(f"X_trn.shape = {X_trn['embs'].shape}, X_tst.shape = {X_tst['embs'].shape}, Z.shape = {Z['embs'].shape}")
        print(f"Y_trn.pos.nnz = {(Y_trn.data == 1).sum()}, Y_trn.neg.nnz = {(Y_trn.data == -1).sum()}, Y_tst.pos.nnz = {(Y_tst.data == 1).sum()}, Y_tst.neg.nnz = {(Y_tst.data == -1).sum()}")
    
    return X_trn, X_tst, Y_trn, Y_tst, Z

def eval_metrics(fm, X_trn, X_tst, Z, qrels_trn_tsv, qrels_tst_tsv, device, batch_size, num_workers, eval_trn):
    
    args = {'batch_size': batch_size, 'depth': 10, 'quiet': True}
    args = namedtuple("args", args.keys())(*args.values())
    
    if eval_trn:
        X_trn_dl = DataLoader(TensorDataset(torch.as_tensor(X_trn['embs'])), 
                              batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    X_tst_dl = DataLoader(TensorDataset(torch.as_tensor(X_tst['embs'])), 
                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    Z_dl = DataLoader(TensorDataset(torch.as_tensor(Z['embs'])), 
                      batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    metrics = {}
    
    fm.eval()
    
    # encode.
    exs_trn, exs_tst, ezs = [], [], []
    with torch.no_grad():
        if eval_trn:
            for xs in tqdm(X_trn_dl, leave=False, desc='Encode X_trn'):
                xs = xs[0].to(device)
                exs_ = fm.encode_x(xs).detach().cpu().numpy()
                exs_trn.append(exs_)
            exs_trn = np.concatenate(exs_trn, axis=0)
        
        for xs in tqdm(X_tst_dl, leave=False, desc='Encode X_tst'):
            xs = xs[0].to(device)
            exs_ = fm.encode_x(xs).detach().cpu().numpy()
            exs_tst.append(exs_)
        exs_tst = np.concatenate(exs_tst, axis=0)
        
        for zs in tqdm(Z_dl, leave=False, desc='Encode Z'):
            zs = zs[0].to(device)
            ezs_ = fm.encode_z(zs).detach().cpu().numpy()
            ezs.append(ezs_)
        ezs = np.concatenate(ezs, axis=0)
    
    # TODO: try normal inner product test performance.
    # exs_tst = exs_tst[:, :-2].copy(order='C')
    # ezs = ezs[:, :-2].copy(order='C')
    
    # retrieval.
    retriever = FaissRetriever(ezs)
    retriever.add(ezs)
    retriever.to_gpu()
    
    # retrieve train
    if eval_trn:
        all_scores, corpus_indices = search_queries(retriever, exs_trn, Z['indices'], args)

        dummy_rank_path = '/tmp/foobar' + uuid.uuid4().hex
        write_ranking(corpus_indices, all_scores, X_trn['indices'], dummy_rank_path)
        to_marco(dummy_rank_path)
        metrics['trn_mrr@10'] = compute_metrics_from_files(qrels_trn_tsv, dummy_rank_path + '.marco')['MRR @10']

        os.remove(dummy_rank_path)
        os.remove(dummy_rank_path + '.marco')
    
    # retrieve test
    all_scores, corpus_indices = search_queries(retriever, exs_tst, Z['indices'], args) # shape=(nq, top_k)
    
    dummy_rank_path = '/tmp/foobar' + uuid.uuid4().hex
    write_ranking(corpus_indices, all_scores, X_tst['indices'], dummy_rank_path)
    to_marco(dummy_rank_path)
    metrics['tst_mrr@10'] = compute_metrics_from_files(qrels_tst_tsv, dummy_rank_path + '.marco')['MRR @10']
    
    os.remove(dummy_rank_path)
    os.remove(dummy_rank_path + '.marco')
    
    fm.train()
    
    return metrics
    
def main():
    args = parse_args()
    if args.identity_factor == 'None':
        args.identity_factor = None
    else:
        args.identity_factor = float(args.identity_factor)
    
    log_handlers = [logging.StreamHandler()]
    
    if args.save_model_dir is not None:
        os.makedirs(os.path.join(args.save_model_dir, 'checkpoints'), exist_ok=True)
        
        with open(os.path.join(args.save_model_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        log_handlers.append(logging.FileHandler(os.path.join(args.save_model_dir, 'trn.log'), mode='w'))
    
    logging.root.handlers = [] # in case someone initialized already.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=log_handlers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_trn, X_tst, Y_trn, Y_tst, Z = load_data(args.emb_dir, args.lbl_dir, args.Y_trn_suffix, args.Y_tst_suffix)
    assert(X_trn['embs'].shape[1] == args.x_emb_dim)
    assert(X_tst['embs'].shape[1] == args.x_emb_dim)
    assert(Z['embs'].shape[1] == args.z_emb_dim)
    
    # pos_weight = (Y_trn.data == -1).sum() / (Y_trn.data == 1).sum() if args.pos_weight is None else args.pos_weight
    
    dset_trn = PecosFMDataset(X_trn['embs'], Z['embs'], Y_trn, n_train_zs=args.n_train_zs)
    dl_trn = DataLoader(dset_trn, batch_size=args.bsize, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, collate_fn=pair_collate_fn)
    
    total_steps = args.nepochs * len(dl_trn)
    
    # load pretrained model
    if args.pretrained_model_dir is not None:
        fm = EmbeddingFactorizationMachineModel.load(args.pretrained_model_dir, args.pretrained_ckpt_path)
        # fm.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, 'fm.pth')))
        if args.pretrained_ckpt_path is None:
            logging.info(f'Loaded pretrained model from {os.path.join(args.pretrained_model_dir, "fm.pth")}.')
        else:
            logging.info(f'Loaded pretrained model from {args.pretrained_ckpt_path}.')
    else:
        fm = EmbeddingFactorizationMachineModel(args.k, X_trn['embs'].shape[1], Z['embs'].shape[1], 
                                                use_bias=args.use_bias, ip=args.ip, use_fm=args.use_fm,
                                                identity_factor=args.identity_factor, tanh=args.tanh, 
                                                q_mode=args.q_mode, normalize=args.normalize)
    fm = fm.to(device)
        
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    steps = 0
    trn_loss = 0
    patience = 0
    start_t = time.time()
    worst_metrics = {'trn_mrr@10': 0, 'tst_mrr@10': 0}
    best_tst_metric = worst_metrics[args.monitor_metric]
    
    # print before training
    if args.eval_zeroth_step:
        fm.eval()
        
        for i, (xs, zs) in enumerate(dl_trn):

            assert(xs.shape[0] * args.n_train_zs == zs.shape[0])
            xs, zs = xs.to(device), zs.to(device)

            scores = fm(xs, zs)
            target = torch.arange(scores.shape[0], device=scores.device, dtype=torch.long)
            target = target * args.n_train_zs

            loss = loss_fn(scores, target)
            trn_loss += loss.item()
        
        tst_metrics = eval_metrics(fm, X_trn, X_tst, Z, args.qrels_trn_tsv, args.qrels_tst_tsv, 
                                   device, args.inference_bsize, args.num_workers, args.eval_trn)
        
        best_tst_metric = tst_metrics[args.monitor_metric]
        
        logging.info(f'Step {steps}: trn_loss = {trn_loss / len(dl_trn):.4f}, ' +
                     f'lr = {np.nan}, ' +
                     ', '.join([f'{k} = {v:.4f}' for k, v in tst_metrics.items()]) +
                     f', best_{args.monitor_metric} = {best_tst_metric:.4f}' +
                     f', time = {time.time() - start_t:.4f}')
        
        trn_loss = 0
        start_t = time.time()
        fm.train()
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, fm.parameters()), 
                                  lr=args.lr, weight_decay=args.reg_rate)
    if args.lr_scheduler is not None:
        if args.lr_scheduler == 'linear':
            # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
            scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
        else:
            raise NotImplementedError
    
    with logging_redirect_tqdm(): # avoid printing breaking logs
        
        pbar = trange(total_steps, leave=True)
        
        for epoch in range(args.nepochs):

            fm.train()

            for i, (xs, zs) in enumerate(dl_trn):

                assert(xs.shape[0] * args.n_train_zs == zs.shape[0])

                steps += 1

                optimizer.zero_grad()
                # xs, zs, ys = xs.to(device), zs.to(device), ys.to(device)
                xs, zs = xs.to(device), zs.to(device)

                scores = fm(xs, zs) # shape = [nx, nz]
                target = torch.arange(scores.shape[0], device=scores.device, dtype=torch.long)
                target = target * args.n_train_zs

                loss = loss_fn(scores, target)
                loss.backward()

                optimizer.step()
                
                if args.lr_scheduler is not None:
                    scheduler.step()

                trn_loss += loss.item()

                if steps % args.eval_interval == 0:
                    tst_metrics = eval_metrics(fm, X_trn, X_tst, Z, args.qrels_trn_tsv, args.qrels_tst_tsv, 
                                               device, args.inference_bsize, args.num_workers, args.eval_trn)

                    if best_tst_metric < tst_metrics[args.monitor_metric]:
                        best_tst_metric = tst_metrics[args.monitor_metric]

                        if args.save_model_dir is not None:
                            torch.save(fm.state_dict(), os.path.join(args.save_model_dir, f'fm.pth'))
                        patience = 0 # reset patience.
                    else:
                        patience += 1

                        if args.auto_stop or ((args.patience is not None) and (patience >= args.patience)):
                            logging.info(f"Early stopping at step {steps}.")
                            break # break from step loop

                    logging.info(f'Step {steps}: trn_loss = {trn_loss / args.eval_interval:.4f}, ' + 
                                 f'lr = {args.lr if args.lr_scheduler is None else scheduler.get_last_lr()[0]:.2E}, ' +
                                 ', '.join([f'{k} = {v:.4f}' for k, v in tst_metrics.items()]) +
                                 f', best_{args.monitor_metric} = {best_tst_metric:.4f}' +
                                 f', time = {time.time() - start_t:.4f}')

                    start_t = time.time()
                    trn_loss = 0
                    
                if steps % args.ckpt_interval == 0:
                    if args.save_model_dir is not None:
                        torch.save(fm.state_dict(), os.path.join(args.save_model_dir, 'checkpoints', f'fm_step={steps}.pth'))
                
                pbar.update(1)

            if args.auto_stop or ((args.patience is not None) and (patience >= args.patience)):
                break # break from training loop

    
if __name__ == '__main__':
    main()
