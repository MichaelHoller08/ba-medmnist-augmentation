import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import models, transforms
from medmnist import PathMNIST, BloodMNIST
import numpy as np
import random
import copy
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import os
from datetime import datetime

# ==========================================
# 0. SETUP & KONFIGURATION
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Nutze Gerät: {DEVICE}")

NUM_SAMPLES_PER_CLASS = 3000
NUM_UNKNOWN_SAMPLES = 1000
UNKNOWN_LABEL = 9 

# ==========================================
# 1. DATEN EINMALIG LADEN
# ==========================================
print("\nLade Datensätze in den Arbeitsspeicher...")
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = PathMNIST(split='train', transform=data_transform, download=False, size=224)
val_dataset = PathMNIST(split='val', transform=data_transform, download=False, size=224)
test_dataset = PathMNIST(split='test', transform=data_transform, download=False, size=224)
blood_train_ds = BloodMNIST(split='train', transform=data_transform, download=False, size=224)

def get_balanced_subset(dataset, n_per_class):
    labels = dataset.labels.flatten()
    indices = []
    for i in np.unique(labels):
        class_indices = np.where(labels == i)[0]
        selected_indices = np.random.choice(class_indices, n_per_class, replace=False)
        indices.extend(selected_indices)
    return Subset(dataset, indices)

class LabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, new_label):
        self.dataset = dataset
        self.new_label = new_label
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, np.array([self.new_label], dtype=np.int64)
    def __len__(self):
        return len(self.dataset)

print("Daten erfolgreich geladen!\n")

# ==========================================
# 2. DIE SCHLEIFE FÜR ÜBER NACHT (3 SEEDS)
# ==========================================
SEEDS_TO_TEST = [42, 100, 150]
#SEEDS_TO_TEST = [1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600]

for current_seed in SEEDS_TO_TEST:
    print("="*60)
    print(f"STARTE EXPERIMENTEN-DURCHLAUF MIT SEED: {current_seed}")
    print("="*60)
    
    set_seed(current_seed)
    
    # Datensätze für diesen Seed neu ziehen
    small_train_ds = get_balanced_subset(train_dataset, NUM_SAMPLES_PER_CLASS)
    blood_indices = np.random.choice(len(blood_train_ds), NUM_UNKNOWN_SAMPLES, replace=False)
    small_blood_ds = Subset(blood_train_ds, blood_indices)
    unknown_ds = LabelWrapper(small_blood_ds, UNKNOWN_LABEL)

    # ------------------------------------------
    # BASELINE EXPERIMENT
    # ------------------------------------------
    print("\n--- BASELINE: Optuna Tuning ---")
    def objective_baseline(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        target_labels = [train_dataset.labels[i][0] for i in small_train_ds.indices]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
        fold_accs = []
        for train_idx, val_idx in skf.split(np.zeros(len(target_labels)), target_labels):
            l_train = DataLoader(Subset(small_train_ds, train_idx), batch_size=batch_size, shuffle=True)
            l_val = DataLoader(Subset(small_train_ds, val_idx), batch_size=batch_size, shuffle=False)
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 9)
            model = model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            best_a = 0
            for _ in range(12):
                model.train()
                for inp, tgt in l_train:
                    inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                    optimizer.zero_grad()
                    loss = criterion(model(inp), tgt)
                    loss.backward()
                    optimizer.step()
                model.eval()
                preds, tgts = [], []
                with torch.no_grad():
                    for inp, tgt in l_val:
                        inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                        preds.extend(torch.max(model(inp), 1)[1].cpu().numpy())
                        tgts.extend(tgt.cpu().numpy())
                a = accuracy_score(tgts, preds)
                if a > best_a: best_a = a
            fold_accs.append(best_a)
        return np.mean(fold_accs)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=current_seed)
    study_base = optuna.create_study(direction="maximize", sampler=sampler)
    study_base.optimize(objective_baseline, n_trials=5) # Für n_trials=5 reicht meistens für Tuning
    best_lr_base = study_base.best_params["lr"]
    best_bs_base = study_base.best_params["batch_size"]

    print("--- BASELINE: Finales Training (80 Epochen) ---")
    final_train_loader = DataLoader(small_train_ds, batch_size=best_bs_base, shuffle=True)
    final_val_loader = DataLoader(val_dataset, batch_size=best_bs_base, shuffle=False)
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 9)
    base_model = base_model.to(DEVICE)
    optimizer_base = optim.Adam(base_model.parameters(), lr=best_lr_base)
    criterion = nn.CrossEntropyLoss()

    best_base_val_acc, best_base_weights = 0.0, None
    for _ in range(80):
        base_model.train()
        for inp, tgt in final_train_loader:
            inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
            optimizer_base.zero_grad()
            loss = criterion(base_model(inp), tgt)
            loss.backward()
            optimizer_base.step()
        base_model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for inp, tgt in final_val_loader:
                inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                preds.extend(torch.max(base_model(inp), 1)[1].cpu().numpy())
                tgts.extend(tgt.cpu().numpy())
        acc = accuracy_score(tgts, preds)
        if acc > best_base_val_acc:
            best_base_val_acc = acc
            best_base_weights = copy.deepcopy(base_model.state_dict())
    base_model.load_state_dict(best_base_weights)

    # Baseline Test
    test_loader = DataLoader(test_dataset, batch_size=best_bs_base, shuffle=False)
    base_model.eval()
    b_preds, b_tgts, b_probs = [], [], []
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
            out = base_model(inp)
            b_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            b_preds.extend(torch.max(out, 1)[1].cpu().numpy())
            b_tgts.extend(tgt.cpu().numpy())

    res_b = {
        "Accuracy": accuracy_score(b_tgts, b_preds)*100,
        "Balanced Acc": balanced_accuracy_score(b_tgts, b_preds)*100,
        "Precision": precision_score(b_tgts, b_preds, average='macro', zero_division=0)*100,
        "Recall": recall_score(b_tgts, b_preds, average='macro', zero_division=0)*100,
        "F1-Score": f1_score(b_tgts, b_preds, average='macro', zero_division=0)*100,
        "AUC-ROC": roc_auc_score(b_tgts, b_probs, multi_class='ovr', average='macro')*100
    }

    # ------------------------------------------
    # STRATEGIE EXPERIMENT
    # ------------------------------------------
    print("\n--- STRATEGIE: Optuna Tuning ---")
    def objective_strategy(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        target_labels = [train_dataset.labels[i][0] for i in small_train_ds.indices]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
        fold_accs = []
        for train_idx, val_idx in skf.split(np.zeros(len(target_labels)), target_labels):
            l_tr_t = Subset(small_train_ds, train_idx)
            l_val = Subset(small_train_ds, val_idx)
            l_tr_c = ConcatDataset([l_tr_t, unknown_ds])
            
            l_comb = DataLoader(l_tr_c, batch_size=batch_size, shuffle=True)
            l_targ = DataLoader(l_tr_t, batch_size=batch_size, shuffle=True)
            l_v = DataLoader(l_val, batch_size=batch_size, shuffle=False)

            # Phase 1
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 10)
            model = model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            best_p1, w_p1 = 0.0, None
            for _ in range(12):
                model.train()
                for inp, tgt in l_comb:
                    inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                    optimizer.zero_grad()
                    loss = criterion(model(inp), tgt)
                    loss.backward()
                    optimizer.step()
                model.eval()
                c, t = 0, 0
                with torch.no_grad():
                    for inp, tgt in l_v:
                        inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                        c += (torch.max(model(inp)[:, :9], 1)[1] == tgt).sum().item()
                        t += tgt.size(0)
                if (c/t) > best_p1:
                    best_p1 = c/t
                    w_p1 = copy.deepcopy(model.state_dict())

            # Phase 2
            if w_p1: model.load_state_dict(w_p1)
            model.fc = nn.Linear(model.fc.in_features, 9)
            model = model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            best_p2 = 0.0
            for _ in range(8):
                model.train()
                for inp, tgt in l_targ:
                    inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                    optimizer.zero_grad()
                    loss = criterion(model(inp), tgt)
                    loss.backward()
                    optimizer.step()
                model.eval()
                preds, tgts = [], []
                with torch.no_grad():
                    for inp, tgt in l_v:
                        inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                        preds.extend(torch.max(model(inp), 1)[1].cpu().numpy())
                        tgts.extend(tgt.cpu().numpy())
                a = accuracy_score(tgts, preds)
                if a > best_p2: best_p2 = a
            fold_accs.append(best_p2)
        return np.mean(fold_accs)

    sampler_s = optuna.samplers.TPESampler(seed=current_seed)
    study_strat = optuna.create_study(direction="maximize", sampler=sampler_s)
    study_strat.optimize(objective_strategy, n_trials=5)
    best_lr_strat = study_strat.best_params["lr"]
    best_bs_strat = study_strat.best_params["batch_size"]

    print("--- STRATEGIE: Finales Training (P1: 50 Epochen, P2: 30 Epochen) ---")
    comb_train_final = ConcatDataset([small_train_ds, unknown_ds])
    f_tr_comb = DataLoader(comb_train_final, batch_size=best_bs_strat, shuffle=True)
    f_tr_targ = DataLoader(small_train_ds, batch_size=best_bs_strat, shuffle=True)
    f_val = DataLoader(val_dataset, batch_size=best_bs_strat, shuffle=False)

    strat_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    strat_model.fc = nn.Linear(strat_model.fc.in_features, 10)
    strat_model = strat_model.to(DEVICE)
    optimizer_strat = optim.Adam(strat_model.parameters(), lr=best_lr_strat)

    # P1 Final
    best_f_p1, w_f_p1 = 0.0, None
    for _ in range(50):
        strat_model.train()
        for inp, tgt in f_tr_comb:
            inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
            optimizer_strat.zero_grad()
            loss = criterion(strat_model(inp), tgt)
            loss.backward()
            optimizer_strat.step()
        strat_model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for inp, tgt in f_val:
                inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                c += (torch.max(strat_model(inp)[:, :9], 1)[1] == tgt).sum().item()
                t += tgt.size(0)
        if (c/t) > best_f_p1:
            best_f_p1 = c/t
            w_f_p1 = copy.deepcopy(strat_model.state_dict())

    # P2 Final
    if w_f_p1: strat_model.load_state_dict(w_f_p1)
    strat_model.fc = nn.Linear(strat_model.fc.in_features, 9)
    strat_model = strat_model.to(DEVICE)
    optimizer_strat = optim.Adam(strat_model.parameters(), lr=best_lr_strat)

    best_f_p2, w_f_p2 = 0.0, None
    for _ in range(30):
        strat_model.train()
        for inp, tgt in f_tr_targ:
            inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
            optimizer_strat.zero_grad()
            loss = criterion(strat_model(inp), tgt)
            loss.backward()
            optimizer_strat.step()
        strat_model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for inp, tgt in f_val:
                inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
                preds.extend(torch.max(strat_model(inp), 1)[1].cpu().numpy())
                tgts.extend(tgt.cpu().numpy())
        a = accuracy_score(tgts, preds)
        if a > best_f_p2:
            best_f_p2 = a
            w_f_p2 = copy.deepcopy(strat_model.state_dict())

    strat_model.load_state_dict(w_f_p2)

    # Strategie Test
    t_loader = DataLoader(test_dataset, batch_size=best_bs_strat, shuffle=False)
    strat_model.eval()
    s_preds, s_tgts, s_probs = [], [], []
    with torch.no_grad():
        for inp, tgt in t_loader:
            inp, tgt = inp.to(DEVICE), tgt.flatten().to(DEVICE).long()
            out = strat_model(inp)
            s_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            s_preds.extend(torch.max(out, 1)[1].cpu().numpy())
            s_tgts.extend(tgt.cpu().numpy())

    res_s = {
        "Accuracy": accuracy_score(s_tgts, s_preds)*100,
        "Balanced Acc": balanced_accuracy_score(s_tgts, s_preds)*100,
        "Precision": precision_score(s_tgts, s_preds, average='macro', zero_division=0)*100,
        "Recall": recall_score(s_tgts, s_preds, average='macro', zero_division=0)*100,
        "F1-Score": f1_score(s_tgts, s_preds, average='macro', zero_division=0)*100,
        "AUC-ROC": roc_auc_score(s_tgts, s_probs, multi_class='ovr', average='macro')*100
    }

    # ------------------------------------------
    # AUTOMATISCHES SPEICHERN IN CSV
    # ------------------------------------------
    experiment_data = {
        "Datum/Uhrzeit": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Seed": current_seed,
        "Target Dataset": "PathMNIST",
        "Unknown Dataset": "BloodMNIST",
        "Samples pro Zielklasse": NUM_SAMPLES_PER_CLASS,
        "Anzahl Unknown Samples": NUM_UNKNOWN_SAMPLES,
        
        "Base_LR": round(best_lr_base, 6),
        "Base_BatchSize": best_bs_base,
        "Strat_LR": round(best_lr_strat, 6),
        "Strat_BatchSize": best_bs_strat,
        
        "Base_Accuracy": round(res_b["Accuracy"], 2),
        "Base_Balanced_Acc": round(res_b["Balanced Acc"], 2),
        "Base_Precision": round(res_b["Precision"], 2),
        "Base_Recall": round(res_b["Recall"], 2),
        "Base_F1_Score": round(res_b["F1-Score"], 2),
        "Base_AUC_ROC": round(res_b["AUC-ROC"], 2),
        
        "Strat_Accuracy": round(res_s["Accuracy"], 2),
        "Strat_Balanced_Acc": round(res_s["Balanced Acc"], 2),
        "Strat_Precision": round(res_s["Precision"], 2),
        "Strat_Recall": round(res_s["Recall"], 2),
        "Strat_F1_Score": round(res_s["F1-Score"], 2),
        "Strat_AUC_ROC": round(res_s["AUC-ROC"], 2),
        
        "Verbesserung_Acc": round(res_s["Accuracy"] - res_b["Accuracy"], 2)
    }

    df = pd.DataFrame([experiment_data])
    csv_filename = "experiment_ergebnisse.csv"
    if not os.path.isfile(csv_filename):
        df.to_csv(csv_filename, index=False, sep=";")
    else:
        df.to_csv(csv_filename, mode='a', header=False, index=False, sep=";")
        
    print(f"\n--- SEED {current_seed} ERFOLGREICH ABGESCHLOSSEN UND GESPEICHERT ---\n")

print("\n==========================================")
print("ALLE EXPERIMENTE FÜR ÜBER NACHT ABGESCHLOSSEN!")
print("==========================================")