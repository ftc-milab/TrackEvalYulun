import time
import traceback
from multiprocessing.pool import Pool
from functools import partial
import os
from . import utils
from .utils import TrackEvalException
from . import _timing
from .metrics import Count
import matplotlib.pyplot as plt

try:
    import tqdm
    TQDM_IMPORTED = True
except ImportError as _:
    TQDM_IMPORTED = False


class Evaluator:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
            _timing.DO_TIMING = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                _timing.DISPLAY_LESS_PROGRESS = True

    @_timing.time
    def evaluate(self, dataset_list, metrics_list, show_progressbar=False):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()
            print('\nEvaluating %i tracker(s) on %i sequence(s) for %i class(es) on %s dataset using the following '
                  'metrics: %s\n' % (len(tracker_list), len(seq_list), len(class_list), dataset_name,
                                     ', '.join(metric_names)))
            # tracker_list: FISH-mario
            # seq_list: FISH
            # class_list: 1
            
            
            # print(type(tracker_list))
            # print(len(tracker_list))
            # print(type(tracker_list[0]))
            # print(tracker_list[0])
            
            # exit()
            # # Evaluate each tracker
            for tracker in tracker_list:
                # if not config['BREAK_ON_ERROR'] then go to next tracker without breaking
                try:
                    # Evaluate each sequence in parallel or in series.
                    # returns a nested dict (res), indexed like: res[seq][class][metric_name][sub_metric field]
                    # e.g. res[seq_0001][pedestrian][hota][DetA]
                    print('\nEvaluating %s\n' % tracker)
                    time_start = time.time()
                    if config['USE_PARALLEL']:
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)

                            with Pool(config['NUM_PARALLEL_CORES']) as pool, tqdm.tqdm(total=len(seq_list)) as pbar:
                                _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                         class_list=class_list, metrics_list=metrics_list,
                                                         metric_names=metric_names)
                                results = []
                                for r in pool.imap(_eval_sequence, seq_list_sorted,
                                                   chunksize=20):
                                    results.append(r)
                                    pbar.update()
                                res = dict(zip(seq_list_sorted, results))

                        else:
                            with Pool(config['NUM_PARALLEL_CORES']) as pool:
                                _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                         class_list=class_list, metrics_list=metrics_list,
                                                         metric_names=metric_names)
                                results = pool.map(_eval_sequence, seq_list)
                                res = dict(zip(seq_list, results))
                    else:
                        res = {}
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)
                            for curr_seq in tqdm.tqdm(seq_list_sorted):
                                
                                res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                              metric_names)
                        else:
                            for curr_seq in sorted(seq_list):
                                print("type(dataset)",type(dataset))
                                exit()
                                res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                              metric_names)
                    
                    ### START OF YULUN CODE

                    for ds, re in res.items():
                        # ds FISH
                        # print("ds",ds)
                        # print("re",re)
                        # exit()
                        for cl, result in re.items():
                            # cl pedestrian
                            # print("cl")
                            # print(cl)
                            # print("type",type(result))
                            # print("keys")
                            # print(result.keys())
                            # print("result['Count']")
                            # print(result['Count'])
                            # print("result['HOTA']")
                            # print(type(result['HOTA']))
                            # print("result['HOTA'] keys")
                            # print(result['HOTA'].keys())
                            # print("result['HOTA']['best_matches']")
                            # print(type(result['HOTA']['best_matches']))
                            # print(len(result['HOTA']['best_matches']))
                            # print(result['HOTA']['best_matches'][0])
                            # exit()
                            with open(os.path.join(dataset.get_output_fol(tracker), f"{ds}-{cl}-bestmatch.txt"), 'w') as bmf:
                                with open(os.path.join(dataset.get_output_fol(tracker), f"{ds}-{cl}-changes.txt"), 'w') as chf:
                                    # best matches
                                    last_matching = {}
                                    last_changed_gt = set([])
                                    last_changed_tr = set([])
                                    lines = {'missed_tr': {}}
                                    trids = set([])
                                    num = 0
                                    plt.figure(figsize=(50, 10))
                                    # print("result['HOTA']['best_matches']",result['HOTA']['best_matches'])
                                    # exit()
                                    for frame, (gt, tr) in enumerate(result['HOTA']['best_matches']):
                                        frame += 1
                                        output = False
                                        matching = {}
                                        change = {}
                                        bmf.write(f'{frame}')

                                        for t in tr:
                                            trids.add(t)

                                        for g, t in zip(gt, tr):
                                            bmf.write(f' {g}-{t}')

                                            matching[g] = t
                                            if g not in last_matching.keys():
                                                output = True
                                                if 'new_gt' not in change.keys():
                                                    change['new_gt'] = []
                                                change['new_gt'].append(g)
                                                if g not in lines.keys():
                                                    lines[g] = []
                                                lines[g].append(([frame], [t]))

                                            elif last_matching[g] != t:
                                                output = True
                                                if 'changed_gt' not in change.keys():
                                                    change['changed_gt'] = []
                                                change['changed_gt'].append(g)
                                                if lines[g][-1][0][-1] != frame - 1:
                                                    lines[g][-1][0].append(frame - 1)
                                                    lines[g][-1][1].append(last_matching[g])
                                                lines[g][-1][0].append(frame)
                                                lines[g][-1][1].append(t)
                                        
                                        bmf.write('\n')

                                        if len(gt) > len(tr):
                                            change['missed_gt'] = set(gt[len(tr):])
                                            if change['missed_gt'] != last_changed_gt:
                                                output = True
                                                for g in change['missed_gt'] - last_changed_gt:
                                                    if lines[g][-1][0][-1] != frame - 1:
                                                        lines[g][-1][0].append(frame - 1)
                                                        lines[g][-1][1].append(last_matching[g])
                                                    lines[g][-1][0].append(frame)
                                                    lines[g][-1][1].append(0)
                                                last_changed_gt = change['missed_gt']
                                        elif len(gt) < len(tr):
                                            change['missed_tr'] = set(tr[len(gt):])
                                            if change['missed_tr'] != last_changed_tr:
                                                output = True
                                                for t in change['missed_tr'] - last_changed_tr:
                                                    if t not in lines['missed_tr'].keys():
                                                        lines['missed_tr'][t] = []
                                                    lines['missed_tr'][t].append([frame])
                                                for t in last_changed_tr - change['missed_tr']:
                                                    if lines['missed_tr'][t][-1][-1] != frame - 1:
                                                        lines['missed_tr'][t][-1].append(frame - 1)
                                                last_changed_tr = change['missed_tr']
                                        
                                        last_matching = matching

                                        if output:
                                            chf.write(f'{frame}')
                                            for k, v in matching.items():
                                                chf.write(f' {k}-{v}')
                                            chf.write('\n')
                                            for k, v in change.items():
                                                chf.write(f'{k}:')
                                                for i in v:
                                                    chf.write(f'{i},')
                                                chf.write(' ')
                                            chf.write('\n')
                                    # colors = ["r", "b", "g", "y", "m", "c", "orange", "pink", "teal", "brown", "purple"]
                                    colors = [(255, 255, 255),  (95,95,95), (0, 0, 0),#white,gray,black\
                                                (0, 0, 255), (0, 127, 0), (255, 0, 0), #red,green,blue
                                                (127, 127, 0), (255, 0, 255), (0, 255, 255), #cyan, magenta, yellow
                                                (64, 64, 159)]
                                    trids = sorted(list(trids))
                                    legends = []
                                    a = 0
                                    for k, v in lines.items():
                                        print(k,v)
                                        exit()
                                        if k == 'missed_tr':
                                            ll = []
                                            for t, l in v.items():
                                                for x in l:
                                                    a, = plt.plot(x, [trids.index(t)] * len(x), color=(1/4, 1/4, 1/4), label='missed', linewidth=0.5)
                                            if a != 0:
                                                ll.append(a)
                                                legends += ll
                                        else:
                                            ll = []
                                            for x, y in v:
                                                if y[-1] == 0:
                                                    yy = list(map(trids.index, y[:-1]))
                                                    a, = plt.plot(x[:-1], yy, color=colors[k % len(colors)], label=f'gt-{k}', linewidth=0.5)
                                                else:
                                                    yy = list(map(trids.index, y))
                                                    a, = plt.plot(x, yy, color=colors[k % len(colors)], label=f'gt-{k}', linewidth=0.5)
                                            if a != 0:
                                                ll.append(a)
                                                legends += ll
                                    
                                    plt.legend(handles=legends)
                                    plt.xlabel("frame")
                                    plt.ylabel("tracker id")
                                    plt.xticks(range(0, len(result['HOTA']['best_matches']) * 51 // 50, len(result['HOTA']['best_matches']) // 50))
                                    plt.yticks(range(len(trids)), list(map(str, trids)))
                                    plt.savefig(os.path.join(dataset.get_output_fol(tracker), f"{ds}-{cl}-idflow.png"), dpi=300)
                                    plt.cla()
                                    plt.figure(figsize=(10, 8))

                    ### END OF YULUN CODE


                    # Combine results over all sequences and then over all classes

                    # collecting combined cls keys (cls averaged, det averaged, super classes)
                    combined_cls_keys = []
                    res['COMBINED_SEQ'] = {}
                    # combine sequences for each class
                    for c_cls in class_list:
                        res['COMBINED_SEQ'][c_cls] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                                        seq_key != 'COMBINED_SEQ'}
                            res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)
                    # combine classes
                    if dataset.should_classes_combine:
                        combined_cls_keys += ['cls_comb_cls_av', 'cls_comb_det_av', 'all']
                        res['COMBINED_SEQ']['cls_comb_cls_av'] = {}
                        res['COMBINED_SEQ']['cls_comb_det_av'] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            cls_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                       res['COMBINED_SEQ'].items() if cls_key not in combined_cls_keys}
                            res['COMBINED_SEQ']['cls_comb_cls_av'][metric_name] = \
                                metric.combine_classes_class_averaged(cls_res)
                            res['COMBINED_SEQ']['cls_comb_det_av'][metric_name] = \
                                metric.combine_classes_det_averaged(cls_res)
                    # combine classes to super classes
                    if dataset.use_super_categories:
                        for cat, sub_cats in dataset.super_categories.items():
                            combined_cls_keys.append(cat)
                            res['COMBINED_SEQ'][cat] = {}
                            for metric, metric_name in zip(metrics_list, metric_names):
                                cat_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                           res['COMBINED_SEQ'].items() if cls_key in sub_cats}
                                res['COMBINED_SEQ'][cat][metric_name] = metric.combine_classes_det_averaged(cat_res)

                    # Print and output results in various formats
                    if config['TIME_PROGRESS']:
                        print('\nAll sequences for %s finished in %.2f seconds' % (tracker, time.time() - time_start))
                    output_fol = dataset.get_output_fol(tracker)
                    tracker_display_name = dataset.get_display_name(tracker)
                    for c_cls in res['COMBINED_SEQ'].keys():  # class_list + combined classes if calculated
                        summaries = []
                        details = []
                        num_dets = res['COMBINED_SEQ'][c_cls]['Count']['Dets']
                        if config['OUTPUT_EMPTY_CLASSES'] or num_dets > 0:
                            for metric, metric_name in zip(metrics_list, metric_names):
                                # for combined classes there is no per sequence evaluation
                                if c_cls in combined_cls_keys:
                                    table_res = {'COMBINED_SEQ': res['COMBINED_SEQ'][c_cls][metric_name]}
                                else:
                                    table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value
                                                 in res.items()}

                                if config['PRINT_RESULTS'] and config['PRINT_ONLY_COMBINED']:
                                    dont_print = dataset.should_classes_combine and c_cls not in combined_cls_keys
                                    if not dont_print:
                                        metric.print_table({'COMBINED_SEQ': table_res['COMBINED_SEQ']},
                                                           tracker_display_name, c_cls)
                                elif config['PRINT_RESULTS']:
                                    metric.print_table(table_res, tracker_display_name, c_cls)
                                if config['OUTPUT_SUMMARY']:
                                    summaries.append(metric.summary_results(table_res))
                                if config['OUTPUT_DETAILED']:
                                    details.append(metric.detailed_results(table_res))
                                if config['PLOT_CURVES']:
                                    metric.plot_single_tracker_results(table_res, tracker_display_name, c_cls,
                                                                       output_fol)
                            if config['OUTPUT_SUMMARY']:
                                utils.write_summary_results(summaries, c_cls, output_fol)
                            if config['OUTPUT_DETAILED']:
                                utils.write_detailed_results(details, c_cls, output_fol)

                    # Output for returning from function
                    output_res[dataset_name][tracker] = res
                    output_msg[dataset_name][tracker] = 'Success'

                except Exception as err:
                    output_res[dataset_name][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dataset_name][tracker] = str(err)
                    else:
                        output_msg[dataset_name][tracker] = 'Unknown error occurred.'
                    print('Tracker %s was unable to be evaluated.' % tracker)
                    print(err)
                    traceback.print_exc()
                    if config['LOG_ON_ERROR'] is not None:
                        with open(config['LOG_ON_ERROR'], 'a') as f:
                            print(dataset_name, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print('\n\n\n', file=f)
                    if config['BREAK_ON_ERROR']:
                        raise err
                    elif config['RETURN_ON_ERROR']:
                        return output_res, output_msg

        return output_res, output_msg


@_timing.time
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
        for gts, ids, matches in zip(raw_data["gt_ids"], raw_data["tracker_ids"], seq_res[cls]['HOTA']['best_matches']):
            if matches == 0:
                continue
            for i in range(len(matches[0])):
                matches[0][i] = gts[matches[0][i]]
            for i in range(len(matches[1])):
                matches[1][i] = ids[matches[1][i]]
    return seq_res
