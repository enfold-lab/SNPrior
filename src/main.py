import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from xgboost import XGBClassifier

from data import PreprocessedGenotypeDataset, prepare_data_paths
from utils import measure_performance, configure_logging
from feature_selection.LD_pruning import LD_prunning


logger = logging.getLogger(__name__)

RANDOM_SEED = 42

RANDOM_SEED_DATA_SPLIT = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4


@measure_performance
def train_ML(X_outer_train, y_outer_train, X_outer_test, inner_splits, method, params):
    inner_train_idx, inner_val_idx = inner_splits[0]
    X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
    y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
    is_binary_classification = len(np.unique(y_outer_train)) == 2

    if method == "SVM":
        base_model = SVC(class_weight='balanced' if is_binary_classification else None, random_state=RANDOM_SEED)
        grid_search = GridSearchCV(
            estimator=base_model, param_grid = params, cv=inner_splits,
            scoring='f1_macro' if is_binary_classification else 'accuracy',
            verbose=1, refit=False, n_jobs=-1
        )
        grid_search.fit(X_outer_train, y_outer_train)
        model = base_model.set_params(**grid_search.best_params_)

    elif method == "SVM_fixed":
        model = SVC(**params, class_weight='balanced' if is_binary_classification else None, random_state=RANDOM_SEED)

    elif method == "SNP-BLUP":
        mu = X_inner_train.mean(axis=0)
        Z_train = X_inner_train - mu
        Z_val = X_inner_val - mu
        Z_test = X_outer_test - mu

        model = RidgeClassifier(random_state=RANDOM_SEED, **params)

        model.fit(Z_train, y_inner_train)
        preds = [model.predict(Z) for Z in (Z_train, Z_val, Z_test)]
        return (*preds, model.get_params())

    elif method == "XGB":
        model = XGBClassifier(**params, 
                              objective='multi:softmax',  # Use softmax for multi-class classification
                              num_class=len(np.unique(y_inner_train)),  # Number of classes
                              use_label_encoder=False,
                              early_stopping_rounds=10,
                              eval_metric='mlogloss',  # Metric used for multiclass classification
                              random_state = RANDOM_SEED)
        
        eval_set = [(X_inner_train, y_inner_train), (X_inner_val, y_inner_val)]
        model.fit(X_inner_train, y_inner_train, eval_set=eval_set, verbose=True)
        
        y_pred_inner_train = model.predict(X_inner_train)
        y_pred_inner_val = model.predict(X_inner_val)
        y_pred_outer_test = model.predict(X_outer_test)

        return (y_pred_inner_train, y_pred_inner_val, y_pred_outer_test, model.get_params())

    elif method == "DT":
        model = DecisionTreeClassifier(random_state = RANDOM_SEED)

    elif method == "RF":
        model = RandomForestClassifier(**params, random_state=RANDOM_SEED)

    elif method == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError(f"Unsupported method: {method}")

    model.fit(X_inner_train, y_inner_train)
    
    y_pred_inner_train = model.predict(X_inner_train)
    y_pred_inner_val = model.predict(X_inner_val)
    y_pred_outer_test = model.predict(X_outer_test)

    return (y_pred_inner_train, y_pred_inner_val, y_pred_outer_test, model.get_params())


def evaluate_performance(y_test, y_pred, label_mapping, save_file_prefix):
    assert len(np.unique(y_test)) == len(label_mapping), "every class must appear in y_test"

    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    conf_matrix = confusion_matrix(y_test, y_pred)

    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    
    class_names = [label_mapping.get(i, f"Class_{i}") for i in range(len(np.unique(y_test)))]
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Creating a dictionary for class-specific metrics
    class_metrics = {
        f"class_{class_name}_accuracy": acc for class_name, precision, recall, f1, acc in zip(class_names, precisions, recalls, f1_scores, class_accuracies)
    }

    # plt.figure(figsize=(12, 10))
    # cmap = sns.light_palette('#355C7D', as_cmap=True)
    # sns.heatmap(
    #     conf_matrix, annot=True, fmt='.0f', cmap=cmap, annot_kws={"size": 12},
    #     xticklabels=class_names, yticklabels=class_names,
    # )
    # plt.xlabel('Predicted', fontsize=20)
    # plt.ylabel('Actual', fontsize=20)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.title('Confusion Matrix', fontsize=20)
    # plt.savefig(f"{save_file_prefix}_confusion_matrix.pdf", format='pdf', dpi = 300)
    # pd.DataFrame(conf_matrix, index=class_names, columns=class_names).to_csv(f"{save_file_prefix}_confusion_matrix.csv")
    # plt.close()

    metrics = {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix,
        **class_metrics
    }

    return metrics


@measure_performance
def select_feature(
    X, y, outer_train_idx, inner_splits, 
    method, n_list, variant_info_df, cache_file_prefix = None, from_cache = False,
): 
    X_outer_train = X[outer_train_idx]
    y_outer_train = y[outer_train_idx]
    inner_train_idx, inner_val_idx = inner_splits[0]

    X_selected_list = []
    if method == "xgb":
        # params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 2, 'gamma': 1, "subsample": 0.8, "colsample_bytree": 0.8, 'reg_lambda': 2, 'reg_alpha': 0.5} # inital trial
        params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1, 'reg_alpha': 0} # current best

        if not (from_cache and os.path.exists(f"{cache_file_prefix}_basic_feature_importance_mean.npy")):
            model = XGBClassifier(**params, 
                                objective='multi:softmax',  # Use softmax for multi-class classification
                                num_class=len(np.unique(y)),  # Number of classes
                                use_label_encoder=False,
                                early_stopping_rounds=10,
                                eval_metric='mlogloss',  # Metric used for multiclass classification
                                random_state = RANDOM_SEED)
            
            X_inner_train, y_inner_train = X_outer_train[inner_train_idx], y_outer_train[inner_train_idx]
            X_inner_val, y_inner_val = X_outer_train[inner_val_idx], y_outer_train[inner_val_idx]

            eval_set = [(X_inner_train, y_inner_train), (X_inner_val, y_inner_val)]
            model.fit(X_inner_train, y_inner_train, eval_set=eval_set, verbose=True)
            logger.info(f" - {method} model train done for feature selection")

            # basic tree based feature importance
            feature_importances_impurity = model.feature_importances_
            if cache_file_prefix is not None:
                np.save(f"{cache_file_prefix}_basic_feature_importance_mean.npy", feature_importances_impurity)
            
        else:
            feature_importances_impurity = np.load(f"{cache_file_prefix}_basic_feature_importance_mean.npy")

        for n in n_list:
            selected_indices = np.argsort(feature_importances_impurity)[-n:][::-1]
            X_selected = X[:, selected_indices]
            X_selected_list.append(X_selected)


    elif method == "rf":
        if len(X.shape) >= 3:
            raise NotImplementedError

        params = {'n_estimators': 500, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
        
        if not (from_cache and os.path.exists(f"{cache_file_prefix}_basic_feature_importance_mean.npy")):
            model = RandomForestClassifier(**params, random_state = RANDOM_SEED)
            model.fit(X_outer_train[inner_train_idx], y_outer_train[inner_train_idx])
            logger.info(f" - {method} model train done for feature selection")

            # basic tree based feature importance
            feature_importances_impurity = model.feature_importances_
            if cache_file_prefix is not None:
                np.save(f"{cache_file_prefix}_basic_feature_importance_mean.npy", feature_importances_impurity)

        else:
            feature_importances_impurity = np.load(f"{cache_file_prefix}_basic_feature_importance_mean.npy")

        for n in n_list:
            selected_indices = np.argsort(feature_importances_impurity)[-n:][::-1]
            X_selected = X[:, selected_indices]
            X_selected_list.append(X_selected)

    elif method == "ld_pruning":
        feature_importances_LD_prunning = LD_prunning(X_outer_train, n_list, variant_info_df)

        for n in n_list:
            selected_indices = np.argsort(feature_importances_LD_prunning)[-n:][::-1]
            X_selected = X[:, selected_indices]
            X_selected_list.append(X_selected)

    else:
        num_snps_before = X.shape[1]
        for n in n_list:
            if method in ["random", "variance", "fst", "af"]:
                if method == "random":
                    rng = np.random.default_rng(seed = RANDOM_SEED)
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    selected_indices = rng.choice(num_snps_before, n, replace=False)
                    boolean_mask[selected_indices] = True

                elif method == "variance":
                    batch_process = num_snps_before > 1000000

                    if batch_process:
                        batch_size = 1000000
                        n_samples, n_snps = X.shape
                        variances = np.zeros((n_snps, )) #np.zeros((n_snps, feature_dim))

                        for start in tqdm(range(0, n_snps, batch_size)):
                            end = min(start + batch_size, n_snps)
                            batch_var = np.var(X_outer_train[:, start:end], axis=0)
                            variances[start:end] = batch_var
                    else:
                        variances = np.var(X_outer_train, axis=0)

                    if cache_file_prefix is not None:
                        np.save(f"{cache_file_prefix}_feature_variance.npy", variances)

                    selected_indices = np.argsort(variances)[-n:]
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True
                elif method == "fst":
                    ## This is ANOVA F-value based implemenation of Fst
                    unique_pops = np.unique(y_outer_train)
                    num_variants = X.shape[1]

                    total_mean_freq = np.nanmean(X_outer_train, axis=0) / 2  # 전체 대립 유전자 빈도 계산

                    # 집단 간 및 집단 내 분산 초기화
                    ss_between = np.zeros(num_variants)
                    ss_within = np.zeros(num_variants)

                    for pop in unique_pops:
                        pop_indices = np.where(y_outer_train == pop)[0]
                        pop_data = X_outer_train[pop_indices, :]

                        pop_mean_freq = np.nanmean(pop_data, axis=0) / 2  # 집단별 평균 대립 유전자 빈도 계산
                        ss_between += len(pop_indices) * (pop_mean_freq - total_mean_freq) ** 2  # 집단 간 분산 계산
                        ss_within += np.nansum(((pop_data / 2) - pop_mean_freq) ** 2, axis=0)  # 집단 내 분산 계산

                    # 자유도 계산
                    df_between = len(unique_pops) - 1
                    df_within = len(y_outer_train) - len(unique_pops)

                    # 평균 제곱 계산
                    ms_between = ss_between / df_between
                    ms_within = ss_within / df_within

                    fst = np.nan_to_num(ms_between / (ms_between + ms_within))
                    selected_indices = np.argsort(fst)[-n:]
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True
                elif method == "af":
                    allele_counts = X_outer_train
                    allele_freqs = np.nanmean(allele_counts, axis=0) / 2.0
                    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
                    selected_indices = np.argsort(mafs)[-n:]

                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True

                X_selected = X[:, boolean_mask]
                num_snps_after = X_selected.shape[1]

                assert boolean_mask.sum() == num_snps_after

            elif method == "mrmr":
                _, y_numeric = np.unique(y_outer_train, return_inverse=True)
                relevance = np.abs(np.corrcoef(X_outer_train, y_numeric, rowvar=False)[-1, :-1])

                selected = []
                remaining = set(range(X.shape[1]))
                corr_matrix = np.abs(np.corrcoef(X_outer_train, rowvar=False))

                while len(selected) < n and remaining:
                    best_score = -np.inf
                    best_feature = None

                    for feature in remaining:
                        redundancy = np.mean([corr_matrix[feature, sel] for sel in selected]) if selected else 0
                        score = relevance[feature] - redundancy

                        if score > best_score:
                            best_score = score
                            best_feature = feature

                    selected.append(best_feature)
                    remaining.remove(best_feature)
                X_selected = X[:, selected]
                
            elif method in ["chi2", "f_classif", "mutual_info_classif"]:
                if method == "chi2":
                    score_fun = chi2
                elif method == "f_classif":
                    score_fun = f_classif
                elif method == "mutual_info_classif":
                    score_fun = mutual_info_classif
                else:
                    raise

                selector = SelectKBest(score_fun, k=n)
                selector.fit(X_outer_train, y_outer_train)
                X_selected = selector.transform(X)

            elif method == "recursive_feature_selection":
                knn = KNeighborsClassifier(n_neighbors = n)
                sfs = SequentialFeatureSelector(knn, n_features_to_select = n)
                sfs.fit(X_outer_train, y_outer_train)

                #sfs.get_support()
                X_selected = sfs.transform(X)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            X_selected_list.append(X_selected)

    return(X_selected_list)


def feature_transform(X_train, X_test, n, method='PCA'):
    if method == 'PCA':
        transformer = PCA(n_components=n, random_state = RANDOM_SEED)
    elif method == 'KernelPCA':
        transformer = KernelPCA(n_components=n, kernel='rbf')  # kernel can be changed based on requirement
    # elif method == 'UMAP':
    #     transformer = UMAP(n_components=n)
    else:
        raise ValueError(f"Unsupported method: {method}")

    transformer.fit(X_train)
    
    X_train_transformed = transformer.transform(X_train)
    # X_val_transformed = transformer.transform(X_val)
    X_test_transformed = transformer.transform(X_test)

    return X_train_transformed, X_test_transformed


def draw_PCA(X, y, file_name):
    pca = PCA(n_components=2, random_state = RANDOM_SEED)
    pca_result = pca.fit_transform(X)

    plt.figure(figsize=(6, 6))

    # Get distinct colors from tab20 and tab10 colormaps
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Dark2').colors) + list(plt.get_cmap('Set1').colors) + list(plt.get_cmap('Set2').colors) + list(plt.get_cmap('Set3').colors)
    if len(np.unique(y)) > len(colors):
        raise ValueError("Not enough distinct colors available for the number of classes.")

    color_map = {label: colors[i] for i, label in enumerate(np.unique(y))}

    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label, 
                    color=color_map[label], 
                    alpha=0.5)
    
    plt.title(f'PCA of {X.shape[1]} SNPs')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best', prop={'size': 5})

    plt.savefig(f"{file_name}_PCA.pdf", dpi=300)
    plt.show()


def draw_tSNE(X, y, file_name):
    tsne = TSNE(n_components=2, verbose=1)
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))

    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label, alpha=0.5)

    ## use other color codes
    # labels_unique = np.unique(y)
    # colors = cm.viridis(np.linspace(0, 1, len(labels_unique)))  # Using viridis colormap
    # for label, color in zip(labels_unique, colors):
    #     indices = np.where(y == label)
    #     plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label, alpha=0.3,
    #                 color=color,
    #                 ) 

    plt.title(f't-SNE of {X.shape[1]} SNPs')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.legend(loc='best', prop={'size': 10})
    plt.legend(loc=1, prop={'size': 5})

    plt.savefig(f"{file_name}_tSNE.pdf", dpi = 300)
    plt.show()  # Optionally show the plot


def select_label(y, y_original, target_label):
    target_label_encoded = np.unique(y[y_original == target_label])[0]
    y_binary = (y == target_label_encoded).astype(int)
    y_original_binary = np.where(y_original == target_label, "target", 'others')

    label_mapping = {0 : "others", 1: "target"}
    
    return y_binary, y_original_binary, label_mapping


def select_and_train(target_feature, save_result_file_name = "results.xlsx"):
    ### arguments -----
    logger.info(f"Start select_and_train function with args: {target_feature}, {save_result_file_name}")

    save_dir = Path("./results")
    cache_dir = Path("./cache")
    plot_dir = save_dir / "plots"

    pre_selection_methods = ["variance", "random", "fst", "af", "ld_pruning"] # "chi2", "f_classif"
    n_pre_select_list = [1000000] #[2000000, 4000000, 8000000, 16000000, 32000000]#
    n_pre_select_goal = 1000000

    select_methods = ["random", "xgb", "rf", "variance", "chi2", "f_classif"] # "fst", "af", Extra-trees, "mutual_info_classif"
    select_feature_from_cache = True
    n_select_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]  #131072
    # n_select_list = [100, 1000, 10000, 100000, 1000000] # PCA
    n_dim_reduce_list = [None]  ## list should always contain None to perform whole feature training after selection # [128, 256, 512, 1024, None]

    ML_models = ["SVM"] #["RF", "XGB", "SNP-BLUP"] # ["DT", "KNN"]

    ## Finding Minimum_SNPss
    # pre_selection_methods = ["None"]
    # n_pre_select_goal = 8192
    # n_select_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    ## Direct training without secondary feature selection
    # pre_selection_methods = ["random"]
    # select_methods = ["random"]
    # n_select_list = [1000000]
    # ML_models = ["SVM_fixed"]

    ## For debugging
    # n_pre_select_list = [200]
    # n_pre_select_goal = 200
    # n_select_list = [64, 128]
    # n_dim_reduce_list = [64, 128, None]


    hyper_params = {
        "SVM": [
            {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},  # grid search params
        ],
        "SVM_fixed": [
            {'C': 0.1, 'kernel': 'linear'},   # mostly optimal
        ],
        "RF": [
            # {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # default
            {'n_estimators': 500, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # current best for 1048576 features
        ],

        "XGB": [
            # {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'gamma': 0, 'subsample': 1, 'colsample_bytree': 1, 'reg_lambda': 1, 'reg_alpha': 0}, #default
            {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1, 'reg_alpha': 0}, # current best for 1048576 features
        ],

        "SNP-BLUP" : [{"alpha" : 1.0}],
    }

    ### code start -----
    feature_data_path, label_file = prepare_data_paths()
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dataset = PreprocessedGenotypeDataset(feature_data_path / target_feature, label_file)
    (X, y_original, y), groups, label_mapping, variant_info_df = dataset.get_data()

    outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, 
                                    random_state=RANDOM_SEED_DATA_SPLIT)
    outer_splits = list(outer_cv.split(X, y, groups))

    result_combined = []
    for outer_fold in range(len(outer_splits)):
        outer_train_idx, outer_test_idx = outer_splits[outer_fold]
        y_outer_train = y[outer_train_idx]
        groups_outer_train = groups[outer_train_idx]
        y_outer_test = y[outer_test_idx]
        groups_outer_test = groups[outer_test_idx]
        logger.info(f"Outer fold {outer_fold}: train n={len(y_outer_train)}, test n={len(y_outer_test)}, train groups={np.unique(groups_outer_train).size}, test groups={np.unique(groups_outer_test).size}")

        inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, 
                                        random_state=RANDOM_SEED_DATA_SPLIT)
        inner_splits = list(inner_cv.split(np.zeros(len(outer_train_idx)), y_outer_train, groups_outer_train))

        for pre_feature_select_method in pre_selection_methods:
            pre_select_cache_file_prefix = cache_dir / (
                f"{X.shape[1]}_seed{RANDOM_SEED}_cv{RANDOM_SEED_DATA_SPLIT}"
                f"_outerfold{outer_fold}"
                f"_{pre_feature_select_method}"
            )
            try:
                X_pre_selected_list, perf_metric_preselect = select_feature(
                    X = X, y = y,
                    outer_train_idx = outer_train_idx, inner_splits = inner_splits,
                    method = pre_feature_select_method, n_list = n_pre_select_list, variant_info_df = variant_info_df,
                    # cache_file_prefix = pre_select_cache_file_prefix,
                ) 
            except Exception as e:
                logger.error(f"While pre_select_feature of {pre_feature_select_method} in outer fold {outer_fold}. {e.__class__.__name__}: {str(e)}")
                continue 

            for X_pre_selected, n_pre_select in zip(X_pre_selected_list, n_pre_select_list):
                if n_pre_select > n_pre_select_goal:
                    try:
                        X_pre_selected_final_list, _ = select_feature(
                            X = X_pre_selected, y = y, 
                            outer_train_idx = outer_train_idx, inner_splits = inner_splits,
                            method = "random", n_list = [n_pre_select_goal], variant_info_df = variant_info_df,
                        ) 
                    except Exception as e:
                        logger.error(f"While random selection after pre_select_feature in outer fold {outer_fold}. {e.__class__.__name__}: {str(e)}")
                        continue 
                    X_pre_selected_final = X_pre_selected_final_list[0]
                    logger.info(f" - Further selecting feature by random from {n_pre_select} to {n_pre_select_goal} variants. X_pre_selected.shape = {X_pre_selected.shape}. X_pre_selected_final.shape = {X_pre_selected_final.shape}")
                else:
                    X_pre_selected_final = X_pre_selected
                logger.info(f" - '{pre_feature_select_method}' feature selection selected {min(n_pre_select, n_pre_select_goal)} variants. X_pre_selected_final.shape = {X_pre_selected_final.shape}. perf_metrics_selection: {perf_metric_preselect}")
            
                for feature_select_method in select_methods:
                    current_loop = {
                        "random_seed": RANDOM_SEED,
                        "random_seed_cv": RANDOM_SEED_DATA_SPLIT,
                        "outer_fold": outer_fold,
                        "outer_train_size": len(outer_train_idx),
                        "outer_test_size": len(outer_test_idx),
                        "pre_select_method": pre_feature_select_method,
                        "n_pre_select": n_pre_select,
                        "n_pre_select_goal": n_pre_select_goal,
                        "select_method": feature_select_method
                    }
                    feature_importance_cache_file_prefix = cache_dir / (
                        f"{X.shape[1]}_seed{RANDOM_SEED}_cv{RANDOM_SEED_DATA_SPLIT}"
                        f"_outerfold{outer_fold}"
                        f"_{pre_feature_select_method}_{n_pre_select}_{n_pre_select_goal}"
                        f"_{feature_select_method}"
                    )

            # for _ in [1]: # toggle for minimum_SNPs
            #     perf_metric_preselect = {}
            #     X_pre_selected_final = X
            #     n_pre_select = n_pre_select_goal
            #     y_backup, y_original_backup = y, y_original
            #     feature_select_method = "xgb"
            #     for class_target in ['ACB', 'STU', 'ITU', 'GIH', 'CHB', 'GBR', 'ASW', 'BEB', 'CDX', 'CEU', 'CHS', 'CLM', 'ESN', 'FIN', 'GWD', 'IBS', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'TSI', 'YRI']:
            #         y, y_original, label_mapping = select_label(y_backup, y_original_backup, target_label = class_target)
            #         y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
            #         current_loop = {"random_seed": RANDOM_SEED, "random_seed_cv": RANDOM_SEED_DATA_SPLIT, "outer_fold": outer_fold, "select_method": feature_select_method, "class_target": class_target}
            #         feature_importance_cache_file_prefix = cache_dir / (
            #             f"{X.shape[1]}_seed{RANDOM_SEED}_cv{RANDOM_SEED_DATA_SPLIT}"
            #             f"_outerfold{outer_fold}_{feature_select_method}_cls_{class_target}"
            #         )

                    logger.info(f"*************** current loop: {current_loop} ***************")

                    try:
                        X_selected_list, perf_metric_select = select_feature(
                            X = X_pre_selected_final, y = y, 
                            outer_train_idx = outer_train_idx, inner_splits = inner_splits,
                            method = feature_select_method, n_list = n_select_list, variant_info_df = variant_info_df,
                            cache_file_prefix = feature_importance_cache_file_prefix, from_cache = select_feature_from_cache
                        )
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while select_feature of {current_loop}. {e.__class__.__name__}: {str(e)}")
                        continue 

                    for X_selected, n_select in zip(X_selected_list, n_select_list):
                        current_loop["select_n"] = n_select

                        logger.info(f" - '{feature_select_method}' feature selection selected {n_select} variants. X_selected.shape = {X_selected.shape}. perf_metrics_selection: {perf_metric_select}")

                        # if feature_select_method == "random":
                        #     try:
                        #         save_file_prefix = save_dir / f"{feature_select_method}_{n_select}"
                        #         draw_PCA(X = X_selected, y = y_original, file_name=save_file_prefix)
                        #         # draw_tSNE(X = X_selected, y = y_original, file_name=save_file_prefix)
                        #     except Exception as e:
                        #         logger.error(f"An unexpected error occurred while draw_PCA or draw_tSNE of {current_loop}. {e.__class__.__name__}: {str(e)}")

                        X_outer_train = X_selected[outer_train_idx]
                        X_outer_test = X_selected[outer_test_idx]
                        
                        for n_dim_reduced in n_dim_reduce_list:
                            if (n_dim_reduced is None): # use whole feature
                                current_loop["n_dim_reduced"] = n_select
                                X_outer_train_reduced, X_outer_test_reduced = X_outer_train, X_outer_test
                                logger.info(f" - Using whole features for training: X_outer_train.shape = {X_outer_train_reduced.shape}, X_outer_test.shape = {X_outer_test_reduced.shape}")
                            else:
                                if (n_dim_reduced < n_select):
                                    current_loop["n_dim_reduced"] = n_dim_reduced
                                    try:
                                        X_outer_train_reduced, X_outer_test_reduced = feature_transform(
                                            X_outer_train, X_outer_test, n = n_dim_reduced
                                        )
                                        logger.info(f" - Reduced to {n_dim_reduced} features using PCA: X_outer_train.shape = {X_outer_train_reduced.shape}, X_outer_test.shape = {X_outer_test_reduced.shape}")
                                    except Exception as e:
                                        logger.error(f"While feature_transform of {current_loop}. {e.__class__.__name__}: {str(e)}")
                                        continue
                                else:
                                    continue
                                
                        
                            for train_model in ML_models:
                                for hyper_param_index, current_hyper_param in enumerate(hyper_params[train_model]):
                                    current_loop["train_model"] = train_model

                                    logger.info(f" - Start {train_model} training: X_outer_train.shape = {X_outer_train_reduced.shape} X_outer_test.shape = {X_outer_test_reduced.shape} with hyper_param {current_hyper_param}")

                                    try:
                                        (y_pred_inner_train, y_pred_inner_val, y_pred_outer_test, train_params), perf_metric_train = train_ML(
                                            X_outer_train = X_outer_train_reduced, y_outer_train = y_outer_train, 
                                            X_outer_test = X_outer_test_reduced,
                                            inner_splits = inner_splits,
                                            method = train_model, params = current_hyper_param,
                                        )
                                    except Exception as e:
                                        logger.error(f"An unexpected error occurred while train_ML of {current_loop}. {e.__class__.__name__}: {str(e)}")
                                        continue

                                    inner_train_idx, inner_val_idx = inner_splits[0]
                                    y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]

                                    plot_file_prefix = plot_dir / (
                                        f"{X.shape[1]}_seed{RANDOM_SEED}_cv{RANDOM_SEED_DATA_SPLIT}"
                                        f"_outerfold{outer_fold}"
                                        f"_{pre_feature_select_method}_{n_pre_select}_{n_pre_select_goal}"
                                        f"_{feature_select_method}_{n_select}_{train_model}"
                                    )
                                    eval_metrics_train = evaluate_performance(y_inner_train, y_pred_inner_train, label_mapping, f"{plot_file_prefix}_train")
                                    eval_metrics_val = evaluate_performance(y_inner_val, y_pred_inner_val, label_mapping, f"{plot_file_prefix}_val")
                                    eval_metrics_test = evaluate_performance(y_outer_test, y_pred_outer_test, label_mapping, f"{plot_file_prefix}_test")
                                    logger.info(f' - Train done with Accuracy: {eval_metrics_test["accuracy"]*100:.4f}%, perf_metrics_train: {perf_metric_train}')


                                    merged_metrics = {
                                        **current_loop,
                                        "hyper_params" : str(current_hyper_param),
                                        "model_params" : str(train_params),
                                        **{f"preselect_{k}": v for k, v in perf_metric_preselect.items()},
                                        **{f"select_{k}": v for k, v in perf_metric_select.items()},
                                        **{f"train_{k}": v for k, v in perf_metric_train.items()},
                                        **{f"testset_{k}": v for k, v in eval_metrics_test.items() if k != 'confusion_matrix'},
                                        **{f"valset_{k}": v for k, v in eval_metrics_val.items() if k != 'confusion_matrix'},
                                        **{f"trainset_{k}": v for k, v in eval_metrics_train.items() if k != 'confusion_matrix'},
                                    }
                                    result_combined.append(merged_metrics.copy())

                                    ## update the dataframe
                                    results_df = pd.DataFrame(result_combined)
                                    results_df.to_excel(save_dir / save_result_file_name, index = False)

def main():
    configure_logging(log_dir="logs")

    global RANDOM_SEED
    input_feature_list =  [
        "merged_support3",    # 43M inputs
        # "merged_support3_variance_1000000_xgb_8192",   # minimum_SNPs
        # "merged_support3_random_1k_seed_42",   # For testing purpose
    ]
    seed_list = [42] #, 919, 1204, 624, 306

    for feature_file in input_feature_list:
        for seed in seed_list:
            RANDOM_SEED = seed
            save_result_file_name = f"{feature_file}_seed_{RANDOM_SEED}_nested_cv_results.xlsx"
            select_and_train(target_feature = feature_file, save_result_file_name = save_result_file_name)

if __name__ == "__main__":
    main()
