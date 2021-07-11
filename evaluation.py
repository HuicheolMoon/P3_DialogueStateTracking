import json
import argparse
from eval_utils import DSTEvaluator


def _evaluation(preds, labels, slot_meta):
    evaluator = DSTEvaluator(slot_meta)
    evaluator.init()
    assert len(preds) == len(labels)
    for k, l in labels.items():
        p = preds.get(k)
        if p is None:
            raise Exception(f"{k} is not in the predictions!")
        evaluator.update(l, p)
    result = evaluator.compute()
    print(result)
    return result


def evaluation(gt_path, pred_path, slot_meta_path):
    slot_meta = json.load(open(slot_meta_path))
    gts = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    eval_results = _evaluation(preds, gts, slot_meta)
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_path', type=str, required=True
    )
    parser.add_argument(
        '--pred_path', type=str, required=True
    )
    parser.add_argument(
        '--slot_meta_path', type=str, default='data/train_dataset/slot_meta.json'
    )
    args = parser.parse_args()
    eval_result = evaluation(args.gt_path, args.pred_path, args.slot_meta_path)
