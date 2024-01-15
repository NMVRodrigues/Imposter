from torchmetrics import MetricCollection, FBetaScore, Accuracy, Precision, Recall, AUROC, CalibrationError
from typing import Dict


def cv_metrics(n_classes: int, device: str) -> Dict[str, MetricCollection]:
    metrics = {
        'train': MetricCollection({
            'Acc': Accuracy(num_classes=n_classes, average='macro', task='multiclass').to(device),
            'F1': FBetaScore(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'Pr': Precision(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'Rc': Recall(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'AUC': AUROC(num_classes=n_classes, average='macro', task='multiclass').to(device),
            'CalErr': CalibrationError(num_classes=n_classes, average='macro', task='multiclass').to(device)
        }),
        'val': MetricCollection({
            'Acc': Accuracy(num_classes=n_classes, average='macro', task='multiclass').to(device),
            'F1': FBetaScore(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'Pr': Precision(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'Rc': Recall(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'AUC': AUROC(num_classes=n_classes, average='macro', task='multiclass').to(device),
            'CalErr': CalibrationError(num_classes=n_classes, average='macro', task='multiclass').to(device)
        }),
        'test': MetricCollection({
            'Acc': Accuracy(num_classes=n_classes, average='macro', task='multiclass').to(device),
            'F1': FBetaScore(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'Pr': Precision(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'Rc': Recall(num_classes=n_classes, average='macro', task='multiclass', beta=1.0).to(device),
            'AUC': AUROC(num_classes=n_classes, average='macro', task='multiclass').to(device),
            'CalErr': CalibrationError(num_classes=n_classes, average='macro', task='multiclass').to(device)
        })
    }

    return metrics
