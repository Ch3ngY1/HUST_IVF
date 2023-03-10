from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np
class LogSummary(object):

    def __init__(self, log_path):

        self.log_path = log_path

    # def write_scalars(self, scalars, names, n_iter, tag=None):
    def write_scalars(self, loss, lr, data, n_iter, tag=None, auc=False):
        metric_names = []
        scalars = []
        if data:
            if auc:
                target_list, predict_list, pred_prob = data
                metric_names += ['auc']
                target_list_onehot = np.eye(2)[target_list]
                scalars += [metrics.roc_auc_score(target_list_onehot, pred_prob)]
            else:
                target_list, predict_list = data

            metric_names += ['recall-{}'.format(cls) for cls in range(2)]
            scalars += [metrics.recall_score(target_list, predict_list, average='macro', labels=[cls])
                        for cls in range(2)]

            metric_names += ['acc']
            scalars += [metrics.accuracy_score(target_list, predict_list)]

            metric_names += ['weighted_f1_score']
            scalars += [metrics.f1_score(target_list, predict_list, average='weighted')]

            # metric_names += ['weighted_f1_score']
            # scalars += [metrics.roc_auc_score(target_list, predict_list, average='weighted')]

        metric_names += ['loss']
        scalars += [loss]

        metric_names += ['learning_rate']
        scalars += [lr]


        writer = SummaryWriter(self.log_path)
        for scalar, name in zip(scalars, metric_names):
            if tag is not None:
                name = '/'.join([tag, name])
                writer.add_scalar(name, scalar, n_iter)
        writer.close()
