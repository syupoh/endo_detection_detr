from numpy import FPE_DIVIDEBYZERO
from tqdm import tqdm
import argparse
import pdb
import json
import os
import cv2
import numpy as np

from syslogger.syslogger import Logger
from datetime import datetime
import inspect
import sys

import shutil

def main_worker(args):
    now = datetime.now()
    curtime = now.isoformat()
    curtime = curtime.replace('-', '')
    curtime = curtime.replace(':', '')[2:13]

    source_name = inspect.getfile(inspect.currentframe())
    source_name = os.path.basename(source_name)
    # sys.stdout = Logger('./log_{0}_{1}.txt'.format(source_name[:-3], curtime))
    # logfile = open('./log_{0}_{1}.txt'.format(source_name[:-3], curtime), 'w')

    evalpath = args.evalpath

    if evalpath.find('labeling_211007_8') > -1:
        dataname = 'labeling_211007_8'
    else:
        dataname = 'labeling_211007'
    # evalpath_after = evalpath[evalpath.find('labeling_'):]
    # dataname = evalpath_after[:evalpath_after.find('.')-2]

    dataroot = '{0}/dataset/endoscopy/data_internal/detection/'.format(
        os.getcwd()[:os.getcwd().find('/python/')]
    )

    datapath = '{0}{1}'.format(
        dataroot, dataname
    )

    testjson_file = '{0}/{1}_test.json'.format(
        datapath, dataname
    )

    prediction_file_name = '{0}/coco_instances_results.json'.format(
        evalpath
    )

    testjson = json.load(open(testjson_file))
    refannotations = testjson['annotations']
    refimages = testjson['images']
    refcategories = testjson['categories']
            # refcategories['id']
            # refcategories['name']

    pred_data_set = json.load(open(prediction_file_name))

#########################
#### for outputpath

    if args.vascular_bleeding:
        accuracyname = 'accuracy_vascular_bleeding'
    elif args.all:
        accuracyname = 'accuracy_all'
    elif args.polyp:
        accuracyname = 'accuracy_polyp'
    else:
        accuracyname = 'accuracy'

    if args.imagesave_thr == 0:
        outputpath = '{0}/{1}.txt'.format(
            evalpath, accuracyname
        ) 
    else:
        outputpath = '{0}/{1}_{2:.02f}.txt'.format(
            evalpath, accuracyname, args.imagesave_thr
        ) 
    f = open(outputpath, 'w')
    f.close()
    if args.imagesave_thr == 0:
        thr_set = np.arange(0.9, 0, -0.05)
    else:
        thr_set = [args.imagesave_thr]
    
    best_f1 = 0
    best_thr = 0

    best_f1_category = []
    best_thr_category = []
    for category in refcategories:
        best_f1_category.append(0)
        best_thr_category.append(0)

    if args.vascular_bleeding and not args.imagesave_thr == 0:
        postfix = 'vb_'
    elif args.all and not args.imagesave_thr == 0:
        postfix = 'all_'
    elif args.polyp and not args.imagesave_thr == 0:
        postfix = 'polyp_'
    else:
        postfix = ''

################################################
################## for new_anno

    print(' generate new_anno with {0}'.format(testjson_file))
    print(' {0}'.format(len(refannotations)))
    new_anno = []
    search_index = 0
    for ik in tqdm(range(len(testjson['images'])), total=len(testjson['images'])):
        temp = {}
        temp['image_id'] = refimages[ik]['id']
        temp['file_name'] = refimages[ik]['file_name']
        temp['category_id'] = []
        for category in refcategories:
            temp['category_id'].append(0)

        for i in range(search_index, len(refannotations)):
            refanno = refannotations[i]
            if temp['image_id'] < refanno['image_id']:
                search_index = i
                break
            elif temp['image_id'] == refanno['image_id']:
                if args.all:
                    for jk in range(len(temp['category_id'])):
                        temp['category_id'][jk] = 1
                if args.vascular_bleeding and refanno['category_id'] == 3:
                    temp['category_id'][0] = 1
                if args.vascular_bleeding and refanno['category_id'] == 1:
                    temp['category_id'][2] = 1
                if args.polyp and refanno['category_id'] < 4:
                    temp['category_id'][0] = 1
                    temp['category_id'][1] = 1
                    temp['category_id'][2] = 1
                temp['category_id'][refanno['category_id']-1] = 1
                search_index = i

        new_anno.append(temp)
        

    # logfile.write('=================\n')
        
    for thr in thr_set:
        print('{0:.02f}'.format(thr))
        # logfile.write('{0:.02f}\n'.format(thr))
        result = []

################################################
################## for new_pred

        print(' generate new_pred with {0}'.format(prediction_file_name))
        print(' {0}'.format(len(pred_data_set)))
        new_pred = []
        search_index = 0
        for ik in tqdm(range(len(testjson['images'])), total=len(testjson['images'])):
            temp = {}
            temp['image_id'] = refimages[ik]['id']
            temp['file_name'] = refimages[ik]['file_name']
            temp['category_id'] = []
            for category in refcategories:
                temp['category_id'].append(0)
            for i in range(search_index, len(pred_data_set)):
                pred_data = pred_data_set[i]
                if temp['image_id'] < pred_data['image_id']:
                    search_index = i
                    break
                elif temp['image_id'] == pred_data['image_id'] and pred_data['score'] > thr:
                    if args.all:
                        for jk in range(len(temp['category_id'])):
                            temp['category_id'][jk] = 1
                    if args.vascular_bleeding and pred_data['category_id'] == 3:
                        temp['category_id'][0] = 1
                    if args.vascular_bleeding and pred_data['category_id'] == 1:
                        temp['category_id'][2] = 1
                    if args.polyp and pred_data['category_id'] < 4:
                        temp['category_id'][0] = 1
                        temp['category_id'][1] = 1
                        temp['category_id'][2] = 1
                    temp['category_id'][pred_data['category_id']-1] = 1
                    search_index = i

            new_pred.append(temp)

        
################################################
################## Evaluation

        print(' Evaluation..')
        for category in refcategories:
            temp = {}
            temp['id'] = category['id']
            temp['name'] = category['name']
            temp['FP'] = 0
            temp['TP'] = 0
            temp['FN'] = 0
            temp['TN'] = 0
            result.append(temp)
            
            if not args.imagesave_thr == 0:
                os.makedirs('{0}/{2}FP_{1:.02f}'.format(evalpath, args.imagesave_thr, postfix), exist_ok=True)
                os.makedirs('{0}/{3}FP_{2:.02f}/{1}'.format(evalpath,
                category['id'],
                args.imagesave_thr,
                postfix
                ), exist_ok=True)

                os.makedirs('{0}/{2}FN_{1:.02f}'.format(evalpath, args.imagesave_thr, postfix), exist_ok=True)
                os.makedirs('{0}/{3}FN_{2:.02f}/{1}'.format(evalpath,
                category['id'],
                args.imagesave_thr,
                postfix
                ), exist_ok=True)

        for i in tqdm(range(len(testjson['images'])), total=len(testjson['images'])):
            for ik, category in enumerate(refcategories):
                if new_pred[i]['category_id'][ik] == new_anno[i]['category_id'][ik]:
                    if new_pred[i]['category_id'][ik] == 1:
                        result[ik]['TP'] += 1
                    elif new_pred[i]['category_id'][ik] == 0:
                        result[ik]['TN'] += 1
                elif new_pred[i]['category_id'][ik] == 1:
                    result[ik]['FP'] += 1
                    if not args.imagesave_thr == 0:
                        shutil.copyfile(
                            '{path}/images/{filename}'.format(
                            path = evalpath,
                            filename = new_pred[i]['file_name']
                            ),
                            '{path}/{postfix}FP_{thr:.02f}/{category}/{filename}'.format(
                            path = evalpath,
                            category = ik+1,
                            filename = new_pred[i]['file_name'],
                            thr = thr,
                            postfix=postfix
                            )
                        )
                elif new_pred[i]['category_id'][ik] == 0:
                    result[ik]['FN'] += 1
                    if not args.imagesave_thr == 0:
                        shutil.copyfile(
                            '{path}/images/{filename}'.format(
                            path = evalpath,
                            filename = new_pred[i]['file_name']
                            ),
                            '{path}/{postfix}FN_{thr:.02f}/{category}/{filename}'.format(
                            path = evalpath,
                            category = ik+1,
                            filename = new_pred[i]['file_name'],
                            thr = thr,
                            postfix=postfix
                            )
                        )

################################################
################## overall        
       
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for result_list in result:
            TP += result_list['TP']
            FP += result_list['FP']
            TN += result_list['TN']
            FN += result_list['FN']

        if not TP + FP == 0:
            precision = (TP/(TP + FP)) * 100
        else :
            precision = 0
        if not TP + FN == 0:
            recall = (TP/(TP + FN)) * 100
        else :
            recall = 0

        if not precision + recall == 0:
            f1score = 2 * (precision * recall)/(precision + recall)
        else :
            f1score = 0
        
        if not (TP + TN + FP + FN) == 0:
            accuracy = (TP + TN)/(TP + TN + FP + FN) * 100
        else:
            accuracy = 0
            

        if best_f1 < f1score:
            best_f1 = f1score
            best_thr = thr

        with open(outputpath, 'a') as result_txt:
            result_txt.write('Thr = {0:.02f}\n'.format(thr))
            result_txt.write('precision/recall/f1score/accuracy\t{0:.02f}/{1:.02f}/{2:0.02f}/{3:0.02f}\n'.format(
                precision, recall, f1score, accuracy
                ))
            result_txt.write('category\tTP/TN/FP/FN\tprecision/recall/f1score/accuracy\n')
                

            # logfile.write('======= TP TP2 =======\n')
            for cindex, category in enumerate(result):
                # logfile.write('{0}\n'.format(category['TP']))
                # logfile.write('{0}\n'.format(category['TP2']))
                if category['TP']+category['FP'] == 0:
                    category_precision = 0
                else:
                    category_precision = category['TP'] / (category['TP']+category['FP']) * 100
                    
                if category['TP']+category['FN'] == 0:
                    category_recall = 0
                else:
                    category_recall = category['TP'] / (category['TP']+category['FN']) * 100
                
                if category_precision+category_recall == 0:
                    category_f1 = 0
                else:
                    category_f1 = 2 * (category_precision * category_recall)/(category_precision + category_recall)
                
                if not (category['TP'] + category['TN'] + category['FP'] + category['FN']) == 0:
                    category_accuracy = \
                        (category['TP'] + category['TN'])/(category['TP'] + category['TN'] + category['FP'] + category['FN']) * 100
                else:
                    category_accuracy = 0
                
                if best_f1_category[cindex]  < category_f1:
                    best_f1_category[cindex] = category_f1
                    best_thr_category[cindex] = thr
                
                result_txt.write('{id}.{name}\t{TP}/{TN}/{FP}/{FN}\t{precision:.02f}/{recall:.02f}/{f1:.02f}/{accuracy:.02f}\n'.format(
                    id = category['id'],
                    name = category['name'],
                    TP = category['TP'],
                    TN = category['TN'],
                    FP = category['FP'],
                    FN = category['FN'],
                    precision = category_precision,
                    recall = category_recall,
                    f1 = category_f1,
                    accuracy = category_accuracy
                ))
            result_txt.write('==============\n')
        
        # logfile.write('=================\n')
    
    if args.imagesave_thr == 0:
        os.rename(outputpath, '{0}_{1:.02f}_{2:.02f}_{3:.02f}_{4:.02f}_{5:.02f}_{6:.02f}.txt'.format(
            outputpath[:-4], best_f1, best_thr,
            best_f1_category[0], best_thr_category[0],
            best_f1_category[2], best_thr_category[2],
            ))
    
                
def get_args():
    parser = argparse.ArgumentParser(description="Detectron2 tutorial")
    
    parser.add_argument(
        '--evalpath', type=str,
        default='./log/_eval/eval_labeling_211007_211007T1803_50k_0.050_evalend')
    parser.add_argument(
        '--imagesave-thr', type=float,
        default=0)
    parser.add_argument('--vascular-bleeding', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--polyp', action='store_true')
    # parser.add_argument('--imagesave', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_worker(args)


if __name__ == '__main__':
    main()