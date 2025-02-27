import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import ConfusionMatrix

def calculate_metrics(module, model, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move o modelo para GPU se disponível
    model.eval()

    steps_outputs_metrics = []
    #all_preds = []
    #all_targets = []

    for images, gt_masks in tqdm(module, desc="Calculando métricas"):
        images, gt_masks = images.to(device), gt_masks.to(device)  # Move os dados para GPU
        
        with torch.no_grad():
            logits = model(images)
        
        #pr_masks = F.softmax(logits, dim=1)
        pr_masks = torch.argmax(logits, dim=1)

        # Armazena as previsões e rótulos verdadeiros para matriz de confusão
        #all_preds.append(pr_masks.view(-1))
        #all_targets.append(gt_masks.view(-1))

        # Calcula TP, FP, FN e TN
        tp, fp, fn, tn = smp.metrics.get_stats(
            gt_masks, pr_masks, mode="multiclass", num_classes=num_classes
        )
        steps_outputs_metrics.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

    # Concatena previsões e rótulos verdadeiros para matriz de confusão
    #all_preds = torch.cat(all_preds)
    #all_targets = torch.cat(all_targets)

    # Concatena os valores e mantém os tensores no dispositivo correto
    tp = torch.cat([x["tp"].to(device) for x in steps_outputs_metrics])
    fp = torch.cat([x["fp"].to(device) for x in steps_outputs_metrics])
    fn = torch.cat([x["fn"].to(device) for x in steps_outputs_metrics])
    tn = torch.cat([x["tn"].to(device) for x in steps_outputs_metrics])

    reductions = ["micro", "macro", "micro-imagewise", "macro-imagewise"]

    for reduction in reductions:
        acuracia = smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction).item()
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction).item()
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction).item()
        print("\n")
        print(f"Redução: {reduction}")
        print(f"Acurácia no conjunto de teste: {acuracia:.4f}")
        print(f"IoU no conjunto de teste: {iou:.4f}")
        print(f"F1 no conjunto de teste: {f1_score:.4f}")
        print("\n")

    # Calcula a matriz de confusão
    #conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    #matrix = conf_matrix(all_preds, all_targets).cpu().numpy()  # Move para CPU para plotagem

    # Plot da matriz de confusão
    #plot_confusion_matrix(matrix, class_labels=[f"Classe {i}" for i in range(num_classes)])

    return acuracia, iou, f1_score

def plot_confusion_matrix(conf_matrix, class_labels):
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Previsão")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()
