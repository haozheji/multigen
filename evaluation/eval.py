from bleu.bleu import Bleu
from meteor.meteor_nltk import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser
import csv
import os
import sys
import json
import random
#reload(sys)
#sys.setdefaultencoding('utf-8')

class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        scores_dict = {}
        #scores_dict["model_key"] = self.model_key
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f"%(m, sc))
                    output.append(sc)
                    scores_dict[m] = str(sc)
            else:
                print("%s: %0.5f"%(method, score))
                output.append(score)
                scores_dict[method] = score


        return output

def eval(sources, references, predictions, lower=False, tokenize=False):
    """
        Given a filename, calculate the metric scores for that prediction file
        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    
    for tup in sources:
        pair = {}
        pair['tokenized_sentence'] = tup
        pairs.append(pair)

    cnt = 0
    for line in references:
        pairs[cnt]['tokenized_question'] = line
        cnt += 1

    output = predictions

    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    ## eval
    #from anlg.evaluation.eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        #res[key] = [pair['prediction']]
        res[key] = pair['prediction']
 
        ## gts 
        gts[key].extend(pair['tokenized_question'])

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()




def read_gt(gt_path):

    target = []
    source = []
    with open(os.path.join(gt_path, "target.csv"), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            target.append(line[1:])

    with open(os.path.join(gt_path, "source.csv"), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            source.append(tuple(line[1:]))

    return source, target

def read_hyp(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append([line.strip()])
    return data


def find_all_keys(k, keys):
    res = []
    cur_id = 0
    while(1):
        try:
            find_id = keys.index(k, cur_id)
            cur_id = find_id + 1
            res.append(find_id)
        except:
            break
    return res


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="anlg, eg")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--extra_ids", default=None)
    args = parser.parse_args()
    
    gt_path = os.path.join("data", args.dataset, "test")
    #args.output_dir = os.path.join("../models", args.dataset, args.output_dir)
    hyp_file = args.output_dir
    
    print("scores: \n")
    
    keys, refs = read_gt(gt_path)
    hypos = read_hyp(hyp_file)

   
    if args.extra_ids is not None:
        total_ids = list(range(0, len(keys)))
        #ids = random.sample(total_ids,10000)
        ids = []
        with open(args.extra_ids, 'r') as f:
            for line in f.readlines():
                ids.append(int(line))
        _keys, _refs, _hypos = [], [], []
        keys_set = set()
        for idx in ids:
            keys_set.add(keys[idx])
        #_ids = random.sample(total_ids, 3000)
        
        _ids = []
        for k in keys_set:
            _ids.extend(find_all_keys(k, keys))

        _ids = list(set(_ids))
        
        
        for i in _ids:
            _keys.append(keys[i])
            _refs.append(refs[i])
            _hypos.append(hypos[i])
        keys, refs, hypos = _keys, _refs, _hypos


    eval(keys, refs, hypos, lower=True, tokenize=True)
