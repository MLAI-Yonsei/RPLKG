import pdb
import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)


    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, error_case=False):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        # pdb.set_trace()
        root = f'/mlainas/KGPrompt_data/imagenet/error_case'
        idx = gt[0] // 2
        y_true_npy = gt.data.cpu().numpy()
        y_pred_npy = pred.data.cpu().numpy()
        y_true = y_true_npy.tolist()
        y_pred = y_pred_npy.tolist()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(y_true)
        self._y_pred.extend(y_pred)
        if error_case:
            return np.where(y_pred_npy != y_true_npy), y_true_npy, y_pred_npy
        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)


    def evaluate(self, class_num=None, sent=None, raw_sent=None):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            min_acc = 100
            min_acc_class = 'None'
            if class_num is None:
                for label in labels:
                    classname = self._lab2cname[label]
                    res = self._per_class_res[label]
                    correct = sum(res)
                    total = len(res)
                    acc = 100.0 * correct / total
                    # added code
                    if acc < min_acc:
                        min_acc = acc
                        min_acc_class = label
                    accs.append(acc)
                    print(
                        f"* class: {label} ({classname})\t"
                        f"total: {total:,}\t"
                        f"correct: {correct:,}\t"
                        f"acc: {acc:.1f}%"
                    )
                mean_acc = np.mean(accs)
                print(f"* average: {mean_acc:.1f}%")
                print("***************")
                print("**  Min Acc  **")
                print("***************")
                print(f'Min Acc: {min_acc}')
                print(f'Min Acc class: {min_acc_class}')
                results["perclass_accuracy"] = mean_acc
            else:
                classname = self._lab2cname[class_num]
                res = self._per_class_res[class_num]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                print(
                    f"* class: {class_num} ({classname}), sent: {raw_sent}\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
                return results, correct, self._correct

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
