# %%
#options: cc200, dosenbach160, aal
p_ROI = "cc200"
p_fold = 10
p_center = "Stanford"
p_mode = "whole"
p_augmentation = False
p_Method = "ASD-DiagNet"
num_repitition = 5
USE_REWEIGHTING = False  # Set to True to enable reweight
latent_sex_leakage_evaluation = False

# FairBatch parameters
alpha = 0.002      # step size
lam = None         # lambda for reweighting

# %%
parameter_list = [p_ROI,p_fold,p_center,p_mode,p_augmentation,p_Method]
print("*****List of patameters****")
print("ROI atlas: ",p_ROI)
print("per Center or whole: ",p_mode)
if p_mode == 'percenter':
    print("Center's name: ",p_center)
print("Method's name: ",p_Method)
if p_Method == "ASD-DiagNet":
    print("Augmentation: ",p_augmentation)


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from sklearn.impute import SimpleImputer
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pyprind
import sys
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn import tree
import functools
import numpy.ma as ma # for masked arrays
import pyprind
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ## Import the phenotypic / summary spreadsheet

# %%
df_pheno = pd.read_csv('./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv')

print("Rows, Cols:", df_pheno.shape)
print("Columns:", list(df_pheno.columns))
print(df_pheno['FILE_ID'].head(10))

# %% [markdown]
# ## Import the preprocessed ABIDE ROI time-series (CC200, C-PAC) data

# %%
data_main_path = './data/functionals/cpac/filt_global/rois_' + p_ROI 
flist = [
    f for f in os.listdir(data_main_path)
    if f.endswith('.1D')
] 
print(len(flist))
print(flist[:10])

# %% [markdown]
# ## Combine and filter dataset

# %%
def get_key(filename):
    f_split = filename.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3]) 
    else:
        key = '_'.join(f_split[0:2])
    return key

flist_norm = [get_key(f) for f in flist]

df_fil = df_pheno[df_pheno['FILE_ID'].isin(flist_norm)].copy()

print("Filtered dataset size:", len(df_fil))

# %% [markdown]
# ## Label distribution

# %%
total = len(df_fil)

dx_counts = df_fil['DX_GROUP'].value_counts()

asd_count = dx_counts[1]
control_count = dx_counts[2]

print("Total subjects:", total)
print("ASD (1) count:", asd_count)
print("Control (2) count:", control_count)

print("\nASD (1) percentage: {:.2f}%".format(100 * asd_count / total))
print("Control (2) percentage: {:.2f}%".format(100 * control_count / total))

# %% [markdown]
# ## Sex distribution

# %%
sex_counts = df_fil['SEX'].value_counts()

male_count = sex_counts[1]
female_count = sex_counts[2]

print("\nMale (1) count:", male_count)
print("Female (2) count:", female_count)

print("\nMale (1) percentage: {:.2f}%".format(100 * male_count / total))
print("Female (2) percentage: {:.2f}%".format(100 * female_count / total))

# %% [markdown]
# ## Sex distribution within each diagnosis

# %%
cross_counts = pd.crosstab(df_fil['SEX'], df_fil['DX_GROUP'])
print(cross_counts)

cross_norm_diag = pd.crosstab(df_fil['SEX'],
                             df_fil['DX_GROUP'],
                             normalize='columns'
                             )
print("\nSex distribution within each diagnosis")
print(cross_norm_diag)

# %% [markdown]
# ## Base rates per sex

# %%
base_rates = pd.crosstab(df_fil['SEX'],
                        df_fil['DX_GROUP'],
                        normalize='index')

print("Base Rates (P(DX_GROUP | SEX))")
print(base_rates)

# %% [markdown]
# ## Data Preprocessing and Attribute Mapping 

# %%
for f in range(len(flist)):
    flist[f] = get_key(flist[f])

df_fil.DX_GROUP = df_fil.DX_GROUP.map({1: 1, 2:0})
# 1 -> ASD, 0 -> Control
df_fil.SEX = df_fil.SEX.map({1: 1, 2:0})
# 1 -> Male, 0 -> Female
print(len(df_fil))

# %%
labels = {}
sensitive_attrs = {}
for row in df_fil.iterrows():
    file_id = row[1]['FILE_ID']
    y_label = row[1]['DX_GROUP']
    sensitive_attr = row[1]['SEX']
    if file_id == 'no_filename':
        continue
    assert(file_id not in labels)
    labels[file_id] = y_label
    assert(file_id not in sensitive_attrs)
    sensitive_attrs[file_id] = sensitive_attr

print("Label counts:")
print(pd.Series(labels).value_counts(dropna=False))

print("\nSensitive counts:")
print(pd.Series(sensitive_attrs).value_counts(dropna=False))

# %% [markdown]
# ## Helper functions for computing correlations

# %%
# Returns 0 (Control) or 1 (ASD)
def get_label(filename):
    assert (filename in labels)
    return labels[filename]

# Returns 0 (Female) or 1 (Male)
def get_sensitive(filename):
    assert (filename in sensitive_attrs)
    return sensitive_attrs[filename]


# loads a subject’s ROI fMRI time series, computes the Pearson correlation matrix across regions, and returns the unique pairwise correlations as a 1D feature vector.
def get_corr_data(filename):
    #print(filename)
    for file in os.listdir(data_main_path):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(data_main_path, file), sep='\t')
            # df contains fmri time series data (number of time points (T) * number of ROIs (R))
            
    with np.errstate(invalid="ignore"):
        #cleaning the correlation matrix
        corr = np.nan_to_num(np.corrcoef(df.T)) #corr contains the 200 * 200 (subjects * subjects) correlation matrix

        # creates a lower-triangular matrix (excluding diagonal)
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))

        # Mask everything except the strict lower triangle
        m = ma.masked_where(mask == 1, mask)

        # Flattens, Removes masked values, Returns a 1D NumPy array
        return ma.masked_where(m, corr).compressed()

    # loads a subject’s ROI fMRI time series, computes the Pearson correlation matrix across regions and returns it
def get_corr_matrix(filename):
    for file in os.listdir(data_main_path):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(data_main_path, file), sep='\t')
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        return corr

def confusion(g_turth,predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth,predictions).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    sensitivity = (tp)/(tp+fn)
    specificty = (tn)/(tn+fp)
    return accuracy,sensitivity,specificty

def get_regs(samplesnames,regnum):
    datas = []
    for sn in samplesnames:
        datas.append(all_corr[sn][0])
    datas = np.array(datas)
    avg=[]
    for ie in range(datas.shape[1]):
        avg.append(np.mean(datas[:,ie]))
    avg=np.array(avg)
    highs=avg.argsort()[-regnum:][::-1]
    lows=avg.argsort()[:regnum][::-1]
    regions=np.concatenate((highs,lows),axis=0)
    return regions


# %% [markdown]
# ## Computing correlations and storing it in a file

# %%
if not os.path.exists('./correlations_file'+p_ROI+'_fairness'+'.pkl'):
    pbar=pyprind.ProgBar(len(flist))
    all_corr = {}
    for f in flist:
      
        lab = get_label(f)
        sensitive = get_sensitive(f)
        all_corr[f] = (get_corr_data(f), lab, sensitive)
        pbar.update()

    print('Corr-computations finished')

    pickle.dump(all_corr, open('./correlations_file'+p_ROI+'_fairness'+'.pkl', 'wb'))
    print('Saving to file finished')

else:
    all_corr = pickle.load(open('./correlations_file'+p_ROI+'_fairness'+'.pkl', 'rb'))

print("Number of subjects:", len(all_corr))
print("Tuple length per subject:", len(all_corr[flist[0]]))

# %% [markdown]
# ##  Evaluating Sex attribute encoded in correlations features using logistic regression

# %%
X_list = []
S_list = []

for fname in all_corr:
    feat, _, sex = all_corr[fname]
    X_list.append(feat)
    S_list.append(sex)

X = np.array(X_list)   # shape: N × D
S = np.array(S_list)   # shape: N

# %%
print(X.shape)
print(S.shape)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

scores = cross_val_score(
    clf,
    X,
    S,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("Sex prediction accuracy per fold:", scores)
print("Mean accuracy:", scores.mean())

# %% [markdown]
# ## Computing eigenvalues and eigenvector

# %%
if p_Method=="ASD-DiagNet":
    eig_data = {}
    pbar = pyprind.ProgBar(len(flist))
    for f in flist:  
        d = get_corr_matrix(f)
        eig_vals, eig_vecs = np.linalg.eig(d)

        for ev in eig_vecs.T:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

        sum_eigvals = np.sum(np.abs(eig_vals))
        # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], np.abs(eig_vals[i])/sum_eigvals)
                     for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        eig_data[f] = {'eigvals':np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals':np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs':[ep[1] for ep in eig_pairs]}
        pbar.update()

# %% [markdown]
# ## Calculating Eros similarity

# %%
def norm_weights(sub_flist):
    num_dim = len(eig_data[flist[0]]['eigvals'])
    norm_weights = np.zeros(shape=num_dim)
    for f in sub_flist:
        norm_weights += eig_data[f]['norm-eigvals'] 
    return norm_weights

def cal_similarity(d1, d2, weights, lim=None):
    res = 0.0
    if lim is None:
        weights_arr = weights.copy()
    else:
        weights_arr = weights[:lim].copy()
        weights_arr /= np.sum(weights_arr)
    for i,w in enumerate(weights_arr):
        res += w*np.inner(d1[i], d2[i])
    return res

# %% [markdown]
# ## Defining dataset class

# %%
class CC200Dataset(Dataset):
    def __init__(self, pkl_filename=None, data=None, samples_list=None, 
                 augmentation=False, aug_factor=1, num_neighbs=5,
                 eig_data=None, similarity_fn=None, verbose=False,regs=None):
        self.regs=regs
        if pkl_filename is not None:
            if verbose:
                print ('Loading ..!', end=' ')
            self.data = pickle.load(open(pkl_filename, 'rb'))
        elif data is not None:
            self.data = data.copy()
            
        else:
            sys.stderr.write('Eigther PKL file or data is needed!')
            return 

        #if verbose:
        #    print ('Preprocess..!', end='  ')
        if samples_list is None:
            self.flist = [f for f in self.data]
        else:
            self.flist = [f for f in samples_list]
        self.labels = np.array([self.data[f][1] for f in self.flist])
        
        current_flist = np.array(self.flist.copy())
        current_lab0_flist = current_flist[self.labels == 0]
        current_lab1_flist = current_flist[self.labels == 1]
        #if verbose:
        #    print(' Num Positive : ', len(current_lab1_flist), end=' ')
        #    print(' Num Negative : ', len(current_lab0_flist), end=' ')
        
        
        if augmentation:
            self.num_data = aug_factor * len(self.flist)
            self.neighbors = {}
            pbar = pyprind.ProgBar(len(self.flist))
            weights = norm_weights(samples_list)#??
            for f in self.flist:
                label = self.data[f][1]
                candidates = (set(current_lab0_flist) if label == 0 else set(current_lab1_flist))
                candidates.remove(f)
                eig_f = eig_data[f]['eigvecs']
                sim_list = []
                for cand in candidates:
                    eig_cand = eig_data[cand]['eigvecs']
                    sim = similarity_fn(eig_f, eig_cand,weights)
                    sim_list.append((sim, cand))
                sim_list.sort(key=lambda x: x[0], reverse=True)
                self.neighbors[f] = [item[1] for item in sim_list[:num_neighbs]]#list(candidates)#[item[1] for item in sim_list[:num_neighbs]]
        
        else:
            self.num_data = len(self.flist)

        
    def __getitem__(self, index):
         # ---------- Normal Sample ----------
        if index < len(self.flist):
            fname = self.flist[index]
            data = self.data[fname][0].copy() #get_corr_data(fname, mode=cal_mode)    
            data = data[self.regs].copy()
            label = (self.labels[index],)
            sensitive = self.data[fname][2]
            return (
                torch.FloatTensor(data), torch.FloatTensor(label), torch.LongTensor([sensitive])
            )
        else:
            # ---------- Augmented Sample ----------
            f1 = self.flist[index % len(self.flist)]
            d1, y1, s1 = self.data[f1]
            d1=d1[self.regs]
            f2 = np.random.choice(self.neighbors[f1])
            d2, y2, _ = self.data[f2]
            d2=d2[self.regs]
            assert y1 == y2
            r = np.random.uniform(low=0, high=1)
            label = (y1,)
            data = r*d1 + (1-r)*d2
            return (torch.FloatTensor(data), torch.FloatTensor(label), torch.LongTensor([s1]))

    def __len__(self):
        return self.num_data

# %% [markdown]
# ## Definig data loader function

# %%
def get_loader(pkl_filename=None, data=None, samples_list=None,
               batch_size=64, 
               num_workers=1, mode='train',
               *, augmentation=False, aug_factor=1, num_neighbs=5,
                 eig_data=None, similarity_fn=None, verbose=False,regions=None, sampler=None, dataset=None, drop_last=False):
    """Build and return data loader."""
    
    if mode != 'train':
        augmentation=False

    shuffle = (mode == 'train') and (sampler is None)


    if dataset is None:
        dataset = CC200Dataset(pkl_filename=pkl_filename, data=data, samples_list=samples_list,
                           augmentation=augmentation, aug_factor=aug_factor, 
                           eig_data=eig_data, similarity_fn=similarity_fn, verbose=verbose,regs=regions)

   

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler, drop_last=drop_last)
  
    return data_loader

# %% [markdown]
# ## Defining Autoencoder class

# %%
class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs=990, 
                 num_latent=200, tied=True,
                 num_classes=2, use_dropout=False):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent
        
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
    
        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)
         
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
        
        if use_dropout:
            self.classifier = nn.Sequential (
                nn.Dropout(p=0.5),
                nn.Linear(self.num_latent, 1),
                
            )
        else:
            self.classifier = nn.Sequential (
                nn.Linear(self.num_latent, 1),
            )
            
         
    def forward(self, x, eval_classifier=False, return_latent=False):
        z = self.fc_encoder(x)
        z = torch.tanh(z)
        if eval_classifier:
            x_logit = self.classifier(z)
        else:
            x_logit = None
        
        if self.tied:
            x_rec = F.linear(z, self.fc_encoder.weight.t())
        else:
            x_rec = self.fc_decoder(z)
            
        if return_latent:
            return x_rec, x_logit, z
        
        return x_rec, x_logit

mtae = MTAutoEncoder()

mtae

# latent extraction function
def extract_latents(model, loader):
    model.eval()

    Z_all = []
    S_all = []

    with torch.no_grad():
        for batch_x, _, batch_sensitive in loader:
            data = batch_x.to(device)
            _, _, z = model(data, eval_classifier=True, return_latent=True)
            Z_all.append(z.cpu().numpy())
            S_all.append(batch_sensitive.numpy())

    Z = np.vstack(Z_all)
    S = np.concatenate(S_all).flatten()

    return Z, S

# %% [markdown]
# ## Compute Fairness (Equalized Odds)

# %%
def compute_fairness_gaps(y_true, y_pred, sensitive):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)

    group_metrics = {}

    for g in [0,1]:

        mask_sens = (sensitive == g)

        if mask_sens.sum() == 0:
            # if groups are not present
            group_metrics[g] = (np.nan,np.nan)
            continue

        yt = y_true[mask_sens]
        yp = y_pred[mask_sens]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn)>0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn)>0 else np.nan

        group_metrics[g] = (tpr, fpr)

    if not (np.isnan(group_metrics[1][0]) or np.isnan(group_metrics[0][0])):
        # absolute difference
        delta_tpr = abs(group_metrics[1][0] - group_metrics[0][0])
    else:
        delta_tpr = np.nan

    if not (np.isnan(group_metrics[1][1]) or np.isnan(group_metrics[0][1])):
        delta_fpr = abs(group_metrics[1][1] - group_metrics[0][1])
    else:
       delta_fpr = np.nan 

    return delta_tpr, delta_fpr
  


# %% [markdown]
# ## Compute Fair Batch

# %%
# fair batch: calculate signed TPR gap and group-wise TPRs
def eo_tpr_gap_signed(y_true, y_pred, sensitive):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive = np.asarray(sensitive).astype(int)

    mask_pos = (y_true==1)

    def tpr_for_group(g):
        m = mask_pos & (sensitive == g)
        if m.sum() == 0:
            return np.nan
        tp = np.sum((y_pred[m] == 1))
        fn = np.sum((y_pred[m] == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan
    
    tpr_female = tpr_for_group(0)
    tpr_male = tpr_for_group(1)

    if np.isnan(tpr_male) or np.isnan(tpr_female):
        return np.nan, (tpr_female, tpr_male)
    
    return (tpr_male - tpr_female), (tpr_female, tpr_male)

from torch.utils.data import WeightedRandomSampler

# FairBatch sampler: calculate sample weights based on lambda and class proportions
def make_fairbatch_eo_sampler(dataset, lam, num_samples=None):
    # dataset: an instance of CC200Dataset
    # lam: lambda parameter for reweighting
    # num_samples: number of samples to draw in an epoch)

    dataset_labels = dataset.labels.astype(int)

    sens = np.array([dataset.data[f][2] for f in dataset.flist], dtype=int)

    idx_y1z0 = np.where((dataset_labels == 1) & (sens == 0))[0]
    idx_y1z1 = np.where((dataset_labels == 1) & (sens == 1))[0]
    idx_y0 = np.where(dataset_labels == 0)[0]

    m = len(dataset_labels)
    m1 = int(np.sum(dataset_labels == 1))
    m0 = int(np.sum(dataset_labels == 0))

    c1 = (m1 / m) if m > 0 else 0.0
    lam = float(np.clip(lam, 0.0, c1))

    mass_y1z0 = lam
    mass_y1z1 = (c1 - lam)
    mass_y0 = (m0 / m) if m0 > 0 else 0.0

    weights = np.zeros(m, dtype=np.float64)

    if len(idx_y1z0) > 0:
        weights[idx_y1z0] = mass_y1z0 / len(idx_y1z0)
    if len(idx_y1z1) > 0:
        weights[idx_y1z1] = mass_y1z1 / len(idx_y1z1)
    if len(idx_y0) > 0:
        weights[idx_y0] = mass_y0 / len(idx_y0)

    if weights.sum() <= 0:
        weights[:] = 1.0 / m
    
    if num_samples is None:
        num_samples = len(dataset)

    sampler = WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=num_samples, replacement=True
                                  )
    
    return sampler, lam, c1
    

# %% [markdown]
# ## Defining training and testing functions

# %%
from torch import device


def train(model, epoch, train_loader, p_bernoulli=None, mode='both', lam_factor=1.0, sex_weight_map=None):
    model.train()
    train_losses = []
    for i,(batch_x,batch_y,batch_sensitive) in enumerate(train_loader):
        # if len(batch_x) != batch_size:
        #     continue
        if p_bernoulli is not None:
            if i == 0:
                p_tensor = torch.ones_like(batch_x).to(device)*p_bernoulli
            rand_bernoulli = torch.bernoulli(p_tensor).to(device)

        data, target = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()

        if mode in ['both', 'ae']:
            if p_bernoulli is not None:
                rec_noisy, _ = model(data*rand_bernoulli, False)
                loss_ae = criterion_ae(rec_noisy, data) / len(batch_x)
            else:
                rec, _ = model(data, False)
                loss_ae = criterion_ae(rec, data) / len(batch_x)

        if mode in ['both', 'clf']:
            rec_clean, logits = model(data, True)
            
            if USE_REWEIGHTING:
                sens = batch_sensitive.to(device).long().view(-1)
                target = target.view(-1)
                logits = logits.view(-1)

                loss_per_sample = criterion_clf(logits, target)

                if sex_weight_map is not None:
                    w = torch.ones_like(loss_per_sample)
                    w[sens == 0] = float(sex_weight_map[0])
                    w[sens == 1] = float(sex_weight_map[1])
                    loss_clf = (loss_per_sample * w).mean()
                else:
                    loss_clf = loss_per_sample.mean()
            else:
                loss_clf = criterion_clf(logits, target)

        if mode == 'both':
            loss_total = loss_ae + lam_factor*loss_clf
            train_losses.append([loss_ae.detach().cpu().numpy(), 
                                 loss_clf.detach().cpu().numpy()])
        elif mode == 'ae':
            loss_total = loss_ae
            train_losses.append([loss_ae.detach().cpu().numpy(), 
                                 0.0])
        elif mode == 'clf':
            loss_total = loss_clf
            train_losses.append([0.0, 
                                 loss_clf.detach().cpu().numpy()])

        loss_total.backward()
        optimizer.step()

    return train_losses       

def test(model, criterion, test_loader, 
         eval_classifier=False, num_batch=None):
    test_loss, n_test, correct = 0.0, 0, 0
    all_predss=[]
    all_sensitive= []
    if eval_classifier:
        y_true, y_pred_prob = [], []
    with torch.no_grad():
        model.eval()
        for i,(batch_x,batch_y,batch_sensitive) in enumerate(test_loader, 1):
            if num_batch is not None:
                if i >= num_batch:
                    continue
            data = batch_x.to(device)
            rec, logits = model(data, eval_classifier)

            test_loss += criterion(rec, data).detach().cpu().numpy() 
            n_test += len(batch_x)
            if eval_classifier:
                proba = torch.sigmoid(logits).detach().cpu().numpy()
                preds = np.ones_like(proba, dtype=np.int32)
                preds[proba < 0.5] = 0
                
                # all_predss.extend(preds)###????
                # y_arr = np.array(batch_y, dtype=np.int32)

                preds = preds.flatten()
                y_arr = batch_y.cpu().numpy().flatten()
                all_predss.extend(preds.tolist())

                sens_arr = batch_sensitive.cpu().numpy().flatten()
                all_sensitive.extend(sens_arr.tolist())
                
                correct += np.sum(preds == y_arr)
                y_true.extend(y_arr.tolist())
                y_pred_prob.extend(proba.tolist())
        mlp_acc,mlp_sens,mlp_spef = confusion(y_true,all_predss)

    return  mlp_acc,mlp_sens,mlp_spef, y_true, all_predss, all_sensitive




# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%

if p_Method == "ASD-DiagNet" and p_mode == "whole":
    
    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ",num_corr)
    
    start =time.time()
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001
    num_epochs = 25

    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    if not augmentation:
        eig_data = None

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr/4)
    print(n_lat)
    start= time.time()

    print('p_bernoulli: ', p_bernoulli)
    print('augmentaiton: ', augmentation, 'aug_factor: ', aug_factor, 
          'num_neighbs: ', num_neighbs, 'lim4sim: ', lim4sim)
    print('use_dropout: ', use_dropout)
    print('fair-batch: alpha', alpha, '\n')


    sim_function = functools.partial(cal_similarity, lim=lim4sim)
    y_arr = np.array([get_label(f) for f in flist])
    flist = np.array(flist)
    kk=0 

    rep_mean_tpr = []
    rep_mean_fpr = []
    rep_mean_acc = []
    rep_mean_spec = []
    rep_mean_sens = []

    for rp in range(num_repitition):
        crossval_res_kol=[]
        crossval_fairness = []
        kf = StratifiedKFold(n_splits=p_fold, random_state=1, shuffle=True)
        np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        y_sensitive = np.array([get_sensitive(f) for f in flist])
        strat_labels = y_arr * 2 + y_sensitive # 0 -> female control, 1 -> male control, 2 -> female ASD, 3 -> male ASD
        for kk,(train_index, test_index) in enumerate(kf.split(flist, strat_labels)):
            train_samples, test_samples = flist[train_index], flist[test_index]

            if(USE_REWEIGHTING):
                # --- sensitive attribute: sex-only weights computed for TRAIN fold ---
                train_sex = np.array([get_sensitive(f) for f in train_samples])
                n_train = len(train_sex)
                n_female = np.sum(train_sex == 0)
                n_male = np.sum(train_sex == 1)

                # avoid division by zero
                eps = 1e-12
                p_female = n_female / (n_train + eps)
                p_male = n_male / (n_train + eps)

                # weights inversely proportional to group frequencies (for reweighting during training)
                w_female = 0.5 / (p_female + eps)
                w_male = 0.5 / (p_male + eps)
                
                sex_weight_map = {0: w_female, 1 : w_male}

                # debugging: print group counts and weights
                print(f"Fold {kk+1} | Train samples: {len(train_samples)} | Female: {n_female} | Male: {n_male} | w_female: {w_female:.3f} | w_male: {w_male:.3f}")
            else:
                sex_weight_map = None

            verbose = (True if (kk == 0) else False)

            regions_inds = get_regs(train_samples,int(num_corr/4))

            num_inpp = len(regions_inds)
            n_lat = int(num_inpp/2)

            # intialize dataset
            fb_dataset = CC200Dataset(data=all_corr, samples_list=train_samples,
                                   augmentation=False, regs=regions_inds)
            
            # Fair batch: initialize lambda
            fb_labels = fb_dataset.labels.astype(int)
            sens = np.array([fb_dataset.data[f][2] for f in fb_dataset.flist])

            m = len(fb_labels)
            mask_pos = (fb_labels == 1)

            n_y1 = np.sum(mask_pos)
            n_y1z0 = np.sum(mask_pos & (sens == 0))

            c1 = n_y1 / m if m > 0 else 0.0
            lam_local = n_y1z0 / m if m > 0 else 0.0
            # Fair batch: initialize lambda
            
            # commented for fair batch implementation (we will implement it inside epoch loop)
            # train_loader=get_loader(data=all_corr, samples_list=train_samples, 
            #                         batch_size=batch_size, mode='train',
            #                         augmentation=augmentation, aug_factor=aug_factor, 
            #                         num_neighbs=num_neighbs, eig_data=eig_data, similarity_fn=sim_function, 
            #                         verbose=verbose,regions=regions_inds)
            
            train_eval_loader=get_loader(data=all_corr, samples_list=train_samples, 
                                   batch_size=batch_size, mode='test', augmentation=False, 
                                   verbose=verbose,regions=regions_inds, drop_last=False)

            test_loader=get_loader(data=all_corr, samples_list=test_samples, 
                                   batch_size=batch_size, mode='test', augmentation=False, 
                                   verbose=verbose,regions=regions_inds, drop_last=False)

            model = MTAutoEncoder(tied=True, num_inputs=num_inpp, num_latent=n_lat, use_dropout=use_dropout)
            model.to(device)
            criterion_ae = nn.MSELoss(reduction='sum')
            if USE_REWEIGHTING:
                criterion_clf = nn.BCEWithLogitsLoss(reduction='none')
            else:
                criterion_clf = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD([{'params': model.fc_encoder.parameters(), 'lr': learning_rate_ae},
                                   {'params': model.classifier.parameters(), 'lr': learning_rate_clf}],
                                  momentum=0.9)
            
            # non-fair batch training loop (commented for fair batch implementation)
            # for epoch in range(1, num_epochs+1):
                # if epoch <= 20:
                #     train_losses = train(model, epoch, train_loader, p_bernoulli, mode='both', sex_weight_map=sex_weight_map)
                # else:
                #     train_losses = train(model, epoch, train_loader, p_bernoulli, mode='clf', sex_weight_map=sex_weight_map)

            # fair batch training loop
            for epoch in range(1, num_epochs+1):

                #outer optimization
                sampler, lam_local, c1 = make_fairbatch_eo_sampler(fb_dataset, lam_local, num_samples=len(fb_dataset))
                train_loader = get_loader(dataset=fb_dataset, batch_size=batch_size, mode='train', sampler=sampler, drop_last=True)
                
                # inner optimization
                if epoch <= 20:
                    train_losses = train(model, epoch, train_loader, p_bernoulli, mode='both', sex_weight_map=sex_weight_map)
                else:
                    train_losses = train(model, epoch, train_loader, p_bernoulli, mode='clf', sex_weight_map=sex_weight_map)

                # fairness evaluation at the end of each epoch
                _, _, _, y_t, y_p, s_t = test(model, criterion_ae, train_eval_loader, eval_classifier=True)

                gap_signed, (tpr_f, tpr_m) = eo_tpr_gap_signed(y_t, y_p, s_t)

                # lamda update
                if not np.isnan(gap_signed):
                    # lam_local += alpha * np.sign(gap_signed)
                    lam_local += alpha * gap_signed
                    lam_local = float(np.clip(lam_local, 0.0, c1))
                
                print(f"[Fold {kk+1} | Epoch {epoch}]"
                      f"λ={lam_local:.4f}/{c1:.4f} | "
                      f"TPR_f={tpr_f:.3f} | TPR_m={tpr_m:.3f} | TPR_gap_signed={gap_signed:.3f}"
                      )    
                
            if latent_sex_leakage_evaluation:
                Z_train, S_train = extract_latents(model, train_loader)
                probe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000))
                ])
                sex_scores = cross_val_score(
                    probe,
                    Z_train,
                    S_train,
                    cv=5,
                    scoring="accuracy",)
                print(f"Fold {kk+1} | Latent sex prediction accuracy (probe) (mean across folds): {sex_scores.mean():.3f}")

            acc, sens, spec, y_true, y_pred, sensitive = test(model, criterion_ae, test_loader, eval_classifier=True)
            crossval_res_kol.append([acc, sens, spec])

            delta_tpr, delta_fpr = compute_fairness_gaps(y_true, y_pred, sensitive)
            crossval_fairness.append([delta_tpr, delta_fpr])

            print( f"Fold {kk+1} | " f"Acc={acc:.3f} | " f"Sens={sens:.3f} | " f"Spec={spec:.3f} | " f"TPR_gap={delta_tpr:.3f} | " f"FPR_gap={delta_fpr:.3f}" )
        
        perf_arr = np.array(crossval_res_kol)
        fair_arr = np.array(crossval_fairness)

        print("performance (mean across folds):)")
        mean_acc_rp = perf_arr[:,0].mean()
        mean_sens_rp = perf_arr[:,1].mean()
        mean_spec_rp = perf_arr[:,2].mean()


        print("Accuracy:", mean_acc_rp)
        print("Sensitivity:", mean_sens_rp)
        print("Specifity:", mean_spec_rp)

        rep_mean_acc.append(mean_acc_rp)
        rep_mean_sens.append(mean_sens_rp)
        rep_mean_spec.append(mean_spec_rp)

        print("\nFairness:")
        mean_tpr_rp = np.nanmean(fair_arr[:,0])
        mean_fpr_rp = np.nanmean(fair_arr[:,1])

        print("Mean TPR Gap:", mean_tpr_rp)
        print("Std TPR Gap:", np.nanstd(fair_arr[:,0]))

        print("Mean FPR Gap:", mean_fpr_rp)
        print("Std FPR Gap:", np.nanstd(fair_arr[:,1]))

        rep_mean_tpr.append(mean_tpr_rp)
        rep_mean_fpr.append(mean_fpr_rp)

        

        # print("averages:")
        # print(np.mean(np.array(crossval_res_kol),axis = 0))
        finish= time.time()

        print(finish-start)

    # final mean of repetition means:
    print("\n\nFinal performance results averaged across repetitions:")
    print("Mean Accuracy across repetitions:", np.mean(rep_mean_acc))
    print("Std Accuracy across repetitions:", np.std(rep_mean_acc))
    print("Mean Sensitivity across repetitions:", np.mean(rep_mean_sens))
    print("Std Sensitivity across repetitions:", np.std(rep_mean_sens))
    print("Mean Specificity across repetitions:", np.mean(rep_mean_spec))
    print("Std Specificity across repetitions:", np.std(rep_mean_spec))

    print("\n\nFinal fairness results averaged across repetitions:")
    print("Mean TPR Gap across repetitions:", np.mean(rep_mean_tpr))
    print("Std TPR Gap across repetitions:", np.std(rep_mean_tpr))
    print("Mean FPR Gap across repetitions:", np.mean(rep_mean_fpr))
    print("Std FPR Gap across repetitions:", np.std(rep_mean_fpr))

    


# %%

if p_Method == "ASD-DiagNet" and p_mode == "percenter":
    num_corr = len(all_corr[flist[0]][0])

    flist = os.listdir(data_main_path)

    for f in range(len(flist)):
        flist[f] = get_key(flist[f])
    
    centers_dict = {}
    for f in flist:
        key = f.split('_')[0]

        if key not in centers_dict:
            centers_dict[key] = []
        centers_dict[key].append(f)

    

    flist = np.array(centers_dict[p_center])
    
    start =time.time()
    #flist = np.array(sorted(os.listdir(data_main_path)))
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001
    num_epochs = 35
    
    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr/4)


    print('p_bernoulli: ', p_bernoulli)
    print('augmentation: ', augmentation, 'aug_factor: ', aug_factor, 
          'num_neighbs: ', num_neighbs, 'lim4sim: ', lim4sim)
    print('use_dropout: ', use_dropout, '\n')


    sim_function = functools.partial(cal_similarity, lim=lim4sim)
    all_rp_res=[]
    y_arr = np.array([get_label(f) for f in flist])

    kk=0 
    crossval_res_kol_kol=[]
    for rp in range(10):
        print("========================")
        crossval_res_kol = []
        start= time.time()
        kf = StratifiedKFold(n_splits=p_fold)
        #np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        for kk,(train_index, test_index) in enumerate(kf.split(flist, y_arr)):
        
            train_samples, test_samples = flist[train_index], flist[test_index]

            verbose = (True if (kk == 0) else False)

            regions_inds = get_regs(train_samples,int(num_corr/4))
            num_inpp = len(regions_inds)
            n_lat = int(num_inpp/2)
            num_inpp = len(regions_inds)
            train_loader=get_loader(data=all_corr, samples_list=train_samples, 
                                    batch_size=batch_size, mode='train',
                                    augmentation=augmentation, aug_factor=aug_factor, 
                                    num_neighbs=num_neighbs, eig_data=eig_data, similarity_fn=sim_function, 
                                    verbose=verbose,regions=regions_inds)

            test_loader=get_loader(data=all_corr, samples_list=test_samples, 
                                   batch_size=batch_size, mode='test', augmentation=False, 
                                   verbose=verbose,regions=regions_inds)

            model = MTAutoEncoder(tied=True, num_inputs=num_inpp, num_latent=n_lat, use_dropout=use_dropout)
            model.to(device)
            criterion_ae = nn.MSELoss(reduction='sum')
            criterion_clf = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD([{'params': model.fc_encoder.parameters(), 'lr': learning_rate_ae},
                                   {'params': model.classifier.parameters(), 'lr': learning_rate_clf}],
                                  momentum=0.9)

            for epoch in range(1, num_epochs+1):
                if epoch <= 20:
                    train_losses = train(model, epoch, train_loader, p_bernoulli, mode='both')
                else:
                    train_losses = train(model, epoch, train_loader, p_bernoulli, mode='clf')


            res_mlp = test(model, criterion_ae, test_loader, eval_classifier=True)
            #print("fold",kk+1,":",test(model, criterion_ae, test_loader, eval_classifier=True))
            crossval_res_kol.append(res_mlp)
        print("Result of repeat ",rp,":")
        print(np.mean(np.array(crossval_res_kol),axis = 0))
        all_rp_res.append(np.mean(np.array(crossval_res_kol),axis = 0))
        finish= time.time()

        print("Running time:",finish-start)
    print("Avergae result of 10 repeats: ",np.mean(np.array(all_rp_res),axis = 0))

# %%
if p_Method != "ASD-DiagNet" and p_mode == "whole":
    
    clf = SVC(gamma = 'auto') if  p_Method == 'SVM' else RandomForestClassifier(n_estimators=100)
    overall_result = []
    for rp in range(10):
        kf = StratifiedKFold(n_splits=p_fold, random_state=1, shuffle=True)
        np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        res = []
        for kk,(train_index, test_index) in enumerate(kf.split(flist, y_arr)):
            train_samples, test_samples = np.array(flist)[train_index], np.array(flist)[test_index]
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            for i in train_samples:
                train_data.append(all_corr[i][0])
                train_labels.append(all_corr[i][1])

            for i in test_samples:
                test_data.append(all_corr[i][0])
                test_labels.append(all_corr[i][1])

            
            clf.fit(train_data,train_labels)
            pr = clf.predict(test_data)
            res.append(confusion(test_labels,pr))
        
        print("repeat: ",rp,np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())   
    print("---------------Result of repeating 10 times-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())

# %%
random.seed(19)
np.random.seed(19)
if p_Method != "ASD-DiagNet" and p_mode == "percenter":
    
    clf = SVC(gamma = 'auto') if  p_Method == 'SVM' else RandomForestClassifier(n_estimators=100)
    overall_result = []
    for rp in range(10):
        kf = StratifiedKFold(n_splits=p_fold, random_state=1, shuffle=True)
        np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        res = []
        for kk,(train_index, test_index) in enumerate(kf.split(flist, y_arr)):
            train_samples, test_samples = np.array(flist)[train_index], np.array(flist)[test_index]
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            for i in train_samples:
                train_data.append(all_corr[i][0])
                train_labels.append(all_corr[i][1])

            for i in test_samples:
                test_data.append(all_corr[i][0])
                test_labels.append(all_corr[i][1])

            clf.fit(train_data,train_labels)
            pr = clf.predict(test_data)
            res.append(confusion(test_labels,pr))
        
        print("repeat: ",rp,np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())   
    print("---------------Result of repeating 10 times for: ",p_center,"-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())

# %%


# %%


# %%



