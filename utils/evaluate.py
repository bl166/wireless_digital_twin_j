import numpy as np
import argparse
import pickle
import glob
import time 
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import sys
# sys.path.append('/root/digital-twin/wireless_digital_twin/')

import tensorflow as tf
print(tf.test.gpu_device_name())

import utilfunc as uf
import datagen as dg
import models_univ as models

def get_ns3_eval_multi(dataset_str, kpi, num_runs=1, verbose=False, plot_axis=None, return_discard=False):
    """
    Check raw data dir. Fetch test original run (deemed as ground truth), 
    test validation run#1 (deemed as ns-3 single-run prediction), and additional test 
    validation runs (to compute ns-3 multi-run prediction). 
      If either ground truth or run#1 has missing flow(s), discard that sample. 
    Otherwise, unless all additional runs contain missing flow(s), keep 
    that sample and fill missing entries by nan. 
      Multi-run prediction is calculated as the nanmean of the multiple ns-3
    runs. It is more accurate and more stable than single run.
    ----
    Input:
        - dataset_str: 
            Dataset string in modeling/training context.
            (Note: You may need to change the raw_data_dir location.)
        - plot_axis: 
            List of axis(es) to visualize the results. If 2 axes are provided, will plot
            true vs multi-run predicted KPI values in the 1st axis, and the multi-run
            performance vs number of runs in the 2nd axis. If only 1 axes, then only the 
            latter is plotted. Default is None, plotting nothing. 
    ----
    Output: None
    """
    num_kpis = 5  
    num_path = 10 ##
    
    raw_data_dir = "/root/digital-twin/ns-allinone-3.35/ns-3.35/iofiles5/"
    dmap = {'WifiGrid': 'wifi-grid', 'Nsfnet': 'NSFNet'}
    dd, rr, ss, others = re.findall( '(.+)R(\d+)S(\d+)(.+)', dataset_str )[0]

    raw_data_dir = os.path.join(raw_data_dir, dmap[dd] , f'routing_{rr}', f'seed_{ss}')
    str_fix = '_'.join([f'{ss[0].upper()}+{ss[1]}' for ss in re.findall('([a-zA-Z]+)([0-9]+)', others)])
    files = [
        raw_data_dir + f'/kpis_{str_fix}.txt',
        raw_data_dir + f'/kpis+val1_{str_fix}.txt'
    ]
    files += sorted([ff for ff in glob.glob(
        raw_data_dir + f'/kpis+val*{str_fix}_IDX+0000-1000.txt'
    ) if 'val1' not in ff and 'val4' not in ff and 'val5' not in ff])
    fhandles = [open(f,'r') for f in files]
    if verbose > 0:
        print(len(fhandles), files)

    kpis_all = []
    disc_idx = []
    for di, flines in enumerate(zip(*[f.readlines() for f in fhandles])):
        flines_kips = []
        discard = {k:False for k in range(len(files))}
        for fi, fl in enumerate(flines):
            kpis = [int(k) for k in fl.replace('\n','').split(',')]
            if len(kpis) == num_path*num_kpis: ##
                flines_kips.append(kpis)
            else:
                discard[fi] = True
                flines_kips.append([np.nan for _ in range(num_path*num_kpis)])
        discard = list(discard.values())
        if (discard[0] or discard[1]):
            disc_idx.append(di)
            continue
        elif all(discard[2:]):
            #continue
            kpis_all.append(flines_kips) ##
        else:
            kpis_all.append(flines_kips)
    kpis_all = np.array(kpis_all)
    [f.close() for f in fhandles]

    ##
    if kpi.lower().startswith('de'): #delay
        kpis = kpis_all[...,2::num_kpis]
    elif kpi.lower().startswith('j'): #jitter
        kpis = kpis_all[...,3::num_kpis]
    elif kpi.lower().startswith('t'): #throughput
        kpis = kpis_all[...,4::num_kpis]
    elif kpi.lower().startswith('dr'):
        tx = kpis_all[...,0::num_kpis]
        rx = kpis_all[...,1::num_kpis]
        if '-bi' in kpi:
            kpis = (tx-rx)/tx > .1
        elif '-rate' in kpi:
            kpis = (tx-rx)/tx
        else:
            kpis = tx-rx
            
    if verbose > 0:
        print(kpis.shape)
    
    ##
    y1,y2 = [],[]
    for n in range(2,kpis.shape[1]+1):
        tt = kpis[:,0,:]
        pp = kpis[:,1:n,:].mean(1)
        mm = np.abs(tt - pp)
        y1.append(np.nanmean(mm))
        y2.append(np.nanstd(mm))
        
        if n==2 and plot_axis is not None and len(plot_axis) == 2:
            plot_axis[0].plot(pp.flatten(), tt.flatten(), '.', ms=1, c='k')
            
    y1 = np.array(y1)
    y2 = np.array(y2)    
    ystr1 = 'Mean: '+' --> '.join([f'{y:.1f}' for y in y1])
    ystr2 = 'SD: '+' --> '.join([f'{y:.1f}' for y in y2])

    if verbose != -1:
        print(f'ns-3 1run: {y1[0]:.6f} {y2[0]:.6f}')
        if num_runs > 1:
            print(f'ns-3 {num_runs}runs: {y1[num_runs-1]:.6f} {y2[num_runs-1]:.6f}') 

    if plot_axis is None:
        if verbose != -1:
            print(ystr1)
            print(ystr2)
    else:
        if len(plot_axis) == 2:
            ax0 = plot_axis[0]
            ax0.plot(pp, tt, '.', ms=1)
            ax0.set_aspect('equal')
            ax0.set_ylabel('True delay')
            ax0.set_xlabel('Predicted delay')
            
        ax1 = plot_axis[-1]
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        x = np.arange(1,kpis.shape[1])
        lns1 = ax1.plot(x, y1, ls='-', lw=1, color='tab:blue', marker='.', label=ystr1)
        lns2 = ax2.plot(x, y2, ls='-', lw=1, color='tab:red', marker='.', label=ystr2)

        ax1.set_ylabel('MAE Mean', color='tab:blue') 
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylabel('MAE SD', color='tab:red') 
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_xlabel('# of validation runs by ns-3')
        ax1.set_xticks(x)

        # added these three lines
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right', bbox_to_anchor=(1, 1.2))
        
    if return_discard:
        return kpis[:,0,:], kpis[:,1:(num_runs+1),:].mean(1), np.array(disc_idx)
    else:
        return kpis[:,0,:], kpis[:,1:(num_runs+1),:].mean(1)
        
        
    
def get_model(ModelClass, config, cvf=0, restore=True):
    hparams = dict(**config['GNN'], **config['LearningParams'])
    model = ModelClass(hparams, train_on=config['Paths']['trainon'])
    mclass = re.findall('\'.+\..+\.(.+)\'', str(model.__class__))[0] #<class 'utils.models.PlanNet'>

    model.build()
    #model.compile()

    # save model
    model_dirname = 'TRG+' + '+'.join(model.train_on) + '_DS+' + '+'.join(
        [os.path.basename(d) for d in config['Paths']['data']]
    )
    if cvf > 0:
        log_path = os.path.join(config['Paths']['logs'][0], f'cv{cvf}')
    else:
        log_path = config['Paths']['logs'][0]
    checkpoint_dir = os.path.join(log_path, mclass, model_dirname)#+"_run3_x")
    checkpoint_path_best = os.path.join(checkpoint_dir, "cp-best-loss.ckpt") #loss/mae
    checkpoint_path_late = os.path.join(checkpoint_dir, "cp-latest.ckpt")
    #csvlog_path = os.path.join(checkpoint_dir, "training.log")
    
    # Restore the checkpointed values
    if restore:
        #model.load_weights(checkpoint_path_best).expect_partial() 
        #print(checkpoint_path_best)
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore(checkpoint_path_best).expect_partial() 
    return model, checkpoint_dir    



## ns3 results
def eval_ns3 (rfile, train_on):
    lab2idx = {
        'delay':2, 'jitter':3, 'throughput':4, 
        'drops':None, 'drop-bi':None, 'drop-rate':None
    }
    n_kpis = 5

    y_ns3 = []
    with open(rfile, 'r') as f:
        for l in f.readlines():
            if 'drop' not in train_on:
                y_ns3 += [float(i) for i in l.replace('\n','').split(',')[lab2idx[train_on]::n_kpis]]
            else:
                tx = [float(i) for i in l.replace('\n','').split(',')[0::n_kpis]]
                rx = [float(i) for i in l.replace('\n','').split(',')[1::n_kpis]]
                if train_on == 'drops':
                    y_ns3 += [t-r for t,r in zip(tx,rx)]
                elif train_on == 'drop-rate':
                    y_ns3 += [(t-r)/(t+1e-10) for t,r in zip(tx,rx)]
                elif train_on == 'drops-bi':
                    y_ns3 += [((t-r)/(t+1e-10))>.1 for t,r in zip(tx,rx)]

    y_ns3 = np.array(y_ns3)
    return y_ns3


def eval_model(model, datagen ):
    y_true = []
    y_pred = []
    ela = 0
    for cnt, (inp, lab) in enumerate(datagen):
        t_start = time.time()
        #y_true.append(lab[model.train_on])
        y_true.append(
            tf.concat([tf.expand_dims(lab[t],1) for t in model.train_on], axis=1)
        )
        y_pred.append(model(inp).numpy().squeeze())
        t_end = time.time()
        ela += (t_end - t_start)
        if not cnt%10:
            print(cnt, 'used time {}s'.format(t_end - t_start), end = '\r')

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_pred, y_true, ela/(cnt+1)


### eval best model
def eval_all_models_under(checkpoint_dir, cvfold=0, phase='validate', label_conversion=None, multi=True, verbose=False):

    ckpt_files = [ff[:-len('.index')] for ff in sorted(glob.glob(checkpoint_dir+'/*.ckpt.index'))]
    str_phase = 'vals' if phase.startswith('val') else phase
    if multi:
        pkl_best = os.path.join(checkpoint_dir, f'mae_best_{str_phase}_multi.pkl')
        res_placeholder = {}
        idx = ('mae','loss')
    else: #legacy
        pkl_best = os.path.join(checkpoint_dir, f'mae_best_{str_phase}.pkl')
        res_placeholder = (np.inf,np.inf)
        idx = (0,1)
        
    if os.path.exists(pkl_best):
        with open(pkl_best, 'rb') as f:
            eval_all = pickle.load(f)
    else:
        eval_all = {os.path.basename(kk):res_placeholder for kk in ckpt_files}
    if verbose:
        print(eval_all)
    
    ##
    cfile = glob.glob(os.path.join(checkpoint_dir ,'*.ini'))[0]
    cfg   = uf.PathConfigParser(cfile).as_dict()
    _, dss = dg.get_data_gens_sets(cfg, cvfold)
    margkw = {
        'hparams': dict(**cfg['GNN'], **cfg['LearningParams']), 
        'train_on': cfg['Paths']['trainon']
    }
    if 'plannetemb' in checkpoint_dir.lower():
        model = models.PlanNetEmb(**margkw)
    elif 'plannet' in checkpoint_dir.lower():
        model = models.PlanNet(**margkw)
    elif 'routenet' in checkpoint_dir.lower():
        model = models.RouteNet(**margkw)
    else:
        raise
    model.build()
    ckpt = tf.train.Checkpoint(model)    
            
    for i,checkpoint_path in enumerate(ckpt_files):
        kk = os.path.basename(checkpoint_path)
        if kk not in eval_all or eval_all[kk] == res_placeholder:
            ckpt.restore(checkpoint_path).expect_partial()
            # runs faster with config below
            model.path_update.unroll = True
            model.path_update.stateful = False

            for cnt, batch in enumerate(dss[phase]):
                t_start = time.time()
                log = {k:v.numpy() for k,v in model.test_step(batch).items()}
                t_end = time.time()
                
                if not multi:
                    log = {idx[0]:log['mae'], idx[1]:log['loss']}

                # accumulate batch results
                if not cnt:
                    eval_all[kk] = log
                else:
                    for lk in log.keys():
                        eval_all[kk][lk] += log[lk]    
                        
                if not cnt%10:
                    print(kk, cnt, 'used time {:4f}ms per step,'.format((t_end - t_start)*1000), 
                          'val_mae:', eval_all[kk][idx[0]]/(cnt+1), end = '\r')                
            # average
            if multi:
                eval_all[kk] = {'val_'+k : v/(cnt+1) for k,v in eval_all[kk].items()}  
            else:
                eval_all[kk] = [eval_all[kk][ii]/(cnt+1) for ii in idx]
            with open(pkl_best, 'wb') as f:
                print()
                pickle.dump(eval_all, f, protocol=pickle.HIGHEST_PROTOCOL)         
        
    with open(pkl_best, 'wb') as f:
        pickle.dump(eval_all, f, protocol=pickle.HIGHEST_PROTOCOL)  
    return eval_all


# python utils/evaluate.py --cdir ./output_univ/cv1/PlanNet/TRG+delay+jitter+throughput+drops_DS+NsfnetR1S1St180Dr100_LR5.00e-04_FromScratch_2; python utils/evaluate.py --cdir ./output_univ/cv2/PlanNet/TRG+delay+jitter+throughput+drops_DS+NsfnetR1S1St180Dr100_LR5.00e-04_FromScratch_2; python utils/evaluate.py --cdir ./output_univ/cv3/PlanNet/TRG+delay+jitter+throughput+drops_DS+NsfnetR1S1St180Dr100_LR5.00e-04_FromScratch_2;
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cdir', type=str, 
        default='./output2/cv2/PlanNet/TRG+delay_DS+NsfnetR1S1St180Dr150_LR5.00e-04_x/'
    )
    parser.add_argument('--multilab', type=bool, default=True)
    args = parser.parse_args()
    #cdir = './output2/cv1/PlanNet/TRG+delay_DS+NsfnetR1S1St180Dr75_LR3.00e-04_x/'
    cvf = re.findall('/cv(\d+)/', args.cdir)
    cvf = int(cvf[0]) if len(cvf) == 1 else 0
    
    lab_conv_func = lambda x: x
    models_performance = eval_all_models_under(
        args.cdir, cvfold=cvf, multi=args.multilab, label_conversion=lab_conv_func
    )
    model_beat = min(
        models_performance, 
        key=lambda key: models_performance[key]['val_mae' if args.multilab else 0]
    )
        
    print('Best val model:\n', os.path.join(args.cdir,model_beat), '\n', models_performance[model_beat])
