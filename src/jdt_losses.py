# Taken from https://github.com/zifuwanggg/JDTLosses/blob/master/losses/jdt_loss.py
# %%

"""
Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels <https://arxiv.org/abs/2302.05666>
Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels <https://arxiv.org/abs/2303.16296>
Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union <https://arxiv.org/abs/2310.19252>
"""


import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric


class JDTLoss(_Loss):

    def __init__(
        self,
        mIoUD=1.0,
        mIoUI=0.0,
        mIoUC=0.0,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        smooth=1e-3,
        threshold=0.01,
        norm=1,
        log_loss=False,
        add_CE=False,
        ignore_index=None,
        class_weights=None,
        active_classes_mode_hard="PRESENT",
        active_classes_mode_soft="ALL",
    ):
        """
        Arguments:
            mIoUD (float): The weight of the loss to optimize mIoUD.
            mIoUI (float): The weight of the loss to optimize mIoUI.
            mIoUC (float): The weight of the loss to optimize mIoUC.
            alpha (float): The coefficient of false positives in the Tversky loss.
            beta (float): The coefficient of false negatives in the Tversky loss.
            gamma (float): When `gamma` > 1, the loss focuses more on
                less accurate predictions that have been misclassified.
            smooth (float): A floating number to avoid `NaN` error.
            threshold (float): The threshold to select active classes.
            norm (int): The norm to compute the cardinality.
            log_loss (bool): Compute the log loss or not.
            ignore_index (int | None): The class index to be ignored.
            class_weights (list[float] | None): The weight of each class.
                If it is `list[float]`, its size should be equal to the number of classes.
            active_classes_mode_hard (str): The mode to compute
                active classes when training with hard labels.
            active_classes_mode_soft (str): The mode to compute
                active classes when training with hard labels.

        Comments:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """
        super().__init__()

        assert mIoUD >= 0 and mIoUI >= 0 and mIoUC >= 0 and alpha >= 0 and beta >= 0 and gamma >= 1 and smooth >= 0 and threshold >= 0
        assert isinstance(norm, int) and norm > 0
        assert ignore_index == None or isinstance(ignore_index, int)
        assert class_weights == None or all((isinstance(w, float)) for w in class_weights)
        assert active_classes_mode_hard in ["ALL", "PRESENT"]
        assert active_classes_mode_soft in ["ALL", "PRESENT"]

        self.mIoUD = mIoUD
        self.mIoUI = mIoUI
        self.mIoUC = mIoUC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.norm = norm
        self.log_loss = log_loss
        self.add_CE = add_CE
        self.ignore_index = ignore_index
        if class_weights == None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights)
        self.active_classes_mode_hard = active_classes_mode_hard
        self.active_classes_mode_soft = active_classes_mode_soft

    def forward(self, logits, label, keep_mask=None, prob_predictions=False):
        """
        Arguments:
            logits (torch.Tensor): Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor):
                If it is hard label, its shape should be (B, D1, D2, ...).
                If it is soft label, its shape should be (B, C, D1, D2, ...).
            keep_mask (torch.Tensor | None):
                If it is `torch.Tensor`,
                    its shape should be (B, D1, D2, ...) and
                    its dtype should be `torch.bool`.
        """
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        logits = logits.view(batch_size, num_classes, -1)
        if prob_predictions:
            prob = logits
        else:
            prob = logits.log_softmax(dim=1).exp()

        if keep_mask != None:
            assert keep_mask.dtype == torch.bool
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        loss = self.forward_loss(prob, label, keep_mask, active_classes_mode)

        if self.add_CE:
            loss = loss + self.add_CE * torch.nn.functional.cross_entropy(prob, label, weight=self.class_weights)

        return loss

    def get_image_class_matrix(self, logits, label, prob_predictions=False):
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long
        logits = logits.view(batch_size, num_classes, -1)
        if prob_predictions:
            prob = logits
        else:
            prob = logits.log_softmax(dim=1).exp()

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        prob_card = torch.norm(prob, p=self.norm, dim=2)
        label_card = torch.norm(label, p=self.norm, dim=2)
        diff_card = torch.norm(prob - label, p=self.norm, dim=2)

        if self.norm > 1:
            prob_card = torch.pow(prob_card, exponent=self.norm)
            label_card = torch.pow(label_card, exponent=self.norm)
            diff_card = torch.pow(diff_card, exponent=self.norm)

        tp = (prob_card + label_card - diff_card) / 2
        fp = prob_card - tp
        fn = label_card - tp

        active_classes = self.compute_active_classes(label, active_classes_mode, (batch_size, num_classes), (2,))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return tversky, active_classes

    def get_mious(self, logits, label, prob_predictions=False):
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long
        logits = logits.view(batch_size, num_classes, -1)
        if prob_predictions:
            prob = logits
        else:
            prob = logits.log_softmax(dim=1).exp()

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        prob_card = torch.norm(prob, p=self.norm, dim=2)
        label_card = torch.norm(label, p=self.norm, dim=2)
        diff_card = torch.norm(prob - label, p=self.norm, dim=2)

        if self.norm > 1:
            prob_card = torch.pow(prob_card, exponent=self.norm)
            label_card = torch.pow(label_card, exponent=self.norm)
            diff_card = torch.pow(diff_card, exponent=self.norm)

        tp = (prob_card + label_card - diff_card) / 2
        fp = prob_card - tp
        fn = label_card - tp

        batch_size, num_classes = prob.shape[:2]
        active_classes = self.compute_active_classes(label, active_classes_mode, num_classes, (0, 2))
        loss_mIoUD = self.forward_loss_mIoUD(tp, fp, fn, active_classes)

        active_classes = self.compute_active_classes(label, active_classes_mode, (batch_size, num_classes), (2,))
        loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(tp, fp, fn, active_classes)

        return 1 - loss_mIoUD, 1 - loss_mIoUI, 1 - loss_mIoUC

    def forward_loss(self, prob, label, keep_mask, active_classes_mode):
        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        prob_card = torch.norm(prob, p=self.norm, dim=2)
        label_card = torch.norm(label, p=self.norm, dim=2)
        diff_card = torch.norm(prob - label, p=self.norm, dim=2)

        if self.norm > 1:
            prob_card = torch.pow(prob_card, exponent=self.norm)
            label_card = torch.pow(label_card, exponent=self.norm)
            diff_card = torch.pow(diff_card, exponent=self.norm)

        tp = (prob_card + label_card - diff_card) / 2
        fp = prob_card - tp
        fn = label_card - tp

        loss = 0
        batch_size, num_classes = prob.shape[:2]
        if self.mIoUD > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, num_classes, (0, 2))
            loss_mIoUD = self.forward_loss_mIoUD(tp, fp, fn, active_classes)
            loss += self.mIoUD * loss_mIoUD

        if self.mIoUI > 0 or self.mIoUC > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, (batch_size, num_classes), (2,))
            loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(tp, fp, fn, active_classes)
            loss += self.mIoUI * loss_mIoUI + self.mIoUC * loss_mIoUC

        return loss

    def compute_active_classes(self, label, active_classes_mode, shape, dim):
        if active_classes_mode == "ALL":
            mask = torch.ones(shape, dtype=torch.bool)
        elif active_classes_mode == "PRESENT":
            mask = torch.amax(label, dim) > self.threshold

        active_classes = torch.zeros(shape, dtype=torch.bool, device=label.device)
        active_classes[mask] = 1

        return active_classes

    def forward_loss_mIoUD(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0.0 * torch.sum(tp)

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_mIoUD = -torch.log(tversky)
        else:
            loss_mIoUD = 1.0 - tversky

        if self.gamma > 1:
            loss_mIoUD **= self.gamma

        if self.class_weights != None:
            loss_mIoUD *= self.class_weights

        loss_mIoUD = loss_mIoUD[active_classes]
        loss_mIoUD = torch.mean(loss_mIoUD)

        return loss_mIoUD

    def forward_loss_mIoUIC(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0.0 * torch.sum(tp), 0.0 * torch.sum(tp)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_matrix = -torch.log(tversky)
        else:
            loss_matrix = 1.0 - tversky

        if self.gamma > 1:
            loss_matrix **= self.gamma

        if self.class_weights != None:
            class_weights = self.class_weights.unsqueeze(0).expand_as(loss_matrix)
            loss_matrix *= class_weights

        loss_matrix *= active_classes
        loss_mIoUI = self.reduce(loss_matrix, active_classes, 1)
        loss_mIoUC = self.reduce(loss_matrix, active_classes, 0)

        return loss_mIoUI, loss_mIoUC

    def reduce(self, loss_matrix, active_classes, dim):
        active_sum = torch.sum(active_classes, dim)
        active_dim = active_sum > 0
        loss = torch.sum(loss_matrix, dim)
        loss = loss[active_dim] / active_sum[active_dim]
        loss = torch.mean(loss)

        return loss


# %%


class SoftCorrectDICEMetric(Metric):

    def __init__(self, average=None):

        super().__init__()
        self.average = average
        self.add_state("loss_mIoUD", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_mIoUI", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_mIoUC", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.loss = JDTLoss(alpha=0.5, beta=0.5)

    def update(self, probs, label):

        mDICED, mDICEI, mDICEC = self.loss.get_mious(probs, label, prob_predictions=True)

        self.loss_mIoUD += mDICED
        self.loss_mIoUI += mDICEI
        self.loss_mIoUC += mDICEC
        self.total += 1

    def compute(self):
        avg_mIoUD = self.loss_mIoUD / self.total
        avg_mIoUI = self.loss_mIoUI / self.total
        avg_mIoUC = self.loss_mIoUC / self.total

        results = {"mIoUD": avg_mIoUD, "mIoUI": avg_mIoUI, "mIoUC": avg_mIoUC}

        if self.average is None:
            return results
        else:
            return results[self.average]


class SoftDICECorrectAccuSemiMetric(Metric):

    def __init__(self, alpha=0.5, beta=0.5, ignore_index=None, smooth=0.0):

        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.norm = 1

    def _compute_confs(self, logits, label, keep_mask=None, prob_predictions=False):
        """
        Arguments:
            logits (torch.Tensor): Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor):
                If it is hard label, its shape should be (B, D1, D2, ...).
                If it is soft label, its shape should be (B, C, D1, D2, ...).
            keep_mask (torch.Tensor | None):
                If it is `torch.Tensor`,
                    its shape should be (B, D1, D2, ...) and
                    its dtype should be `torch.bool`.
        """
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        logits = logits.view(batch_size, num_classes, -1)
        if prob_predictions:
            prob = logits
        else:
            prob = logits.log_softmax(dim=1).exp()

        if keep_mask != None:
            assert keep_mask.dtype == torch.bool
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
        else:
            label = label.view(batch_size, num_classes, -1)

        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        prob_card = torch.norm(prob, p=self.norm, dim=2)
        label_card = torch.norm(label, p=self.norm, dim=2)
        diff_card = torch.norm(prob - label, p=self.norm, dim=2)

        if self.norm > 1:
            prob_card = torch.pow(prob_card, exponent=self.norm)
            label_card = torch.pow(label_card, exponent=self.norm)
            diff_card = torch.pow(diff_card, exponent=self.norm)

        tp = (prob_card + label_card - diff_card) / 2
        fp = prob_card - tp
        fn = label_card - tp

        batch_size, num_classes = prob.shape[:2]

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)

        return tp, fp, fn

    def update(self, probs, label, keep_mask=None):
        tp, fp, fn = self._compute_confs(probs, label, keep_mask, prob_predictions=True)
        self.fp = fp + self.fp
        self.tp = tp + self.tp
        self.fn = fn + self.fn

    def compute(self):
        tp, fp, fn = (self.tp, self.fp, self.fn)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.mean(tversky)


# %%
if __name__ == "__main__":
    c_size = 5
    chunks = 100

    A = torch.zeros(1, 2, 50, 50)
    A[:, 0] = 0.00
    A[:, 0, :25, :25] = 1.0

    A[:, 1] = 1.0
    A[:, 1, :25, :25] = 0.0

    B = torch.zeros(1, 2, 50, 50)
    B[:, 0, :25, :25] = 1.0
    B[:, 1] = 1.0
    B[:, 1, :25, :25] = 0.0

    # A = torch.rand(c_size*chunks, 10, 50, 50)**5
    # B = torch.rand(c_size*chunks, 10, 50, 50)**5

    # A = torch.nn.functional.softmax(A, dim=1)
    # B = torch.nn.functional.softmax(B, dim=1)

    l = JDTLoss(mIoUD=1.0, mIoUI=0.0, mIoUC=0.0, alpha=0.5, beta=0.5, active_classes_mode_soft="ALL")

    A_c = torch.chunk(A, chunks=chunks, dim=0)
    B_c = torch.chunk(B, chunks=chunks, dim=0)

    print(1 - l(torch.logit(A), B))

    metric = SoftCorrectDICEMetric()

    metric.update(A, B)
    print(metric.compute())
    metric.reset()

    for a_c, b_c in zip(A_c, B_c):
        metric.update(a_c, b_c)
    print(metric.compute())
