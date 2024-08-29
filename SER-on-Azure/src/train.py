from typing import Dict, List
import argparse
import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve, LearningCurveDisplay, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import azureml.mlflow
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("-q", "--parquet-file", type=str)
    parser.add_argument("-w", "--workspace-name", type=str)
    parser.add_argument("-s", "--subscription-id", type=str)
    parser.add_argument("-g", "--resource-group-name", type=str)
    parser.add_argument("-i", "--app-id", type=str)
    parser.add_argument("-c", "--client-secret", type=str)
    parser.add_argument("-t", "--tenant", type=str)
    parser.add_argument("-x", "--experiment-name", type=str)
    parser.add_argument("-n", "--model-name", type=str)
    # parse args
    args = parser.parse_args()

    # return args
    return args


def standard_scale(df: pd.DataFrame):
    sscaler = StandardScaler(copy=True)
    return sscaler.fit_transform(df)


def minmax_scale(df: pd.DataFrame):
    mmscaler = MinMaxScaler(copy=True)
    return mmscaler.fit_transform(df)


def plot_emotion_distribution(df: pd.DataFrame, plot_size=(10, 8)):
    emotions, count = np.unique(df['emotion'].values, return_counts=True)

    fig, ax = plt.subplots(figsize=plot_size)
    plt.bar(x=range(len(emotions)), height=count)
    plt.xticks(
        ticks=range(len(emotions)),
        labels=[e for e in emotions],
    )
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.close(fig)
    
    return fig


def scale(X: pd.DataFrame):
    sscaler = StandardScaler(copy=True)
    mmscaler = MinMaxScaler(copy=True)
    return {
        "standard": sscaler.fit_transform(X),
        "minmax": mmscaler.fit_transform(X),
    }


def split(features, labels, test_size=.2, random_state=0):
    """A function to split and return a feature matrix."""
    return sklearn.model_selection.train_test_split(features, labels, test_size=test_size, random_state=random_state)


def split2dict(X_train, X_test, y_train, y_test):
    return {
        "train": {
            "X": X_train,
            "y": y_train,
        },
        "test": {
            "X": X_test,
            "y": y_test,
        },
    }


def train(nn_models: List[MLPClassifier], splits: Dict[str, Dict[str, pd.DataFrame]]):
    nnet_scores = {}
    for name, split in zip(["standard", "minmax"], [splits["standard"], splits["minmax"]]):
        print(f"{name.upper()}")
        nnet_scores[name] = []
        for m in nn_models:
            m.fit(split["train"]["X"], split["train"]["y"])
            score = m.score(split["test"]["X"], split["test"]["y"])
    
            m_name = type(m).__name__
            nnet_scores[name].append((m_name, f"{100*score:.2f}%"))
    return nnet_scores


def grid_search(splits: Dict[str, Dict[str, pd.DataFrame]], split_name: str = "standard"):
    mlp = MLPClassifier(
        batch_size=32,
        random_state=0,
    )
    # reduced to best found, just to save compute time -- see notebook for original search
    parameters = {
        "hidden_layer_sizes": [(360,)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [1e-3],
        "epsilon": [1e-8],
        "learning_rate": ["adaptive"],
    }
    gs = GridSearchCV(
        mlp,
        parameters,
        cv=5,
        n_jobs=4,
    )
    gs.fit(splits[split_name]["train"]["X"], splits[split_name]["train"]["y"])
    return gs


def stratifiedkfold(mlp_final: MLPClassifier, target: pd.core.series.Series, splits: Dict[str, Dict[str, pd.DataFrame]], split_name: str = "standard"):
    skfold = StratifiedKFold(
        n_splits=min(target.value_counts().values), # 46
        shuffle=True,
        random_state=0,
    )
    kfold_scores = []
    for train_indices, test_indices in skfold.split(splits[split_name]["train"]["X"], splits[split_name]["train"]["y"]):
        mlp_final.fit(
            splits[split_name]["train"]["X"][train_indices],
            splits[split_name]["train"]["y"][train_indices],
        )
        kfold_scores.append(mlp_final.score(
            splits[split_name]["train"]["X"][test_indices],
            splits[split_name]["train"]["y"][test_indices]),
        )
    return kfold_scores


def score(mlp_final: MLPClassifier, splits:Dict[str, Dict[str, pd.DataFrame]], split_name: str = "standard"):
    y_pred = mlp_final.predict(splits[split_name]["test"]["X"])
    return {
        "accuracy": accuracy_score(splits[split_name]["test"]["y"], y_pred),
        "f1": f1_score(splits[split_name]["test"]["y"], y_pred, average="macro"),
        "precision": precision_score(splits[split_name]["test"]["y"], y_pred, average="macro"),
        "recall": recall_score(splits[split_name]["test"]["y"], y_pred, average="macro"),
    }


def generate_learning_scaling_curves(target: pd.core.series.Series, mlp_final: MLPClassifier, splits:Dict[str, Dict[str, pd.DataFrame]], split_name: str = "standard"):
    params = {
        "X": splits[split_name]["train"]["X"],
        "y": splits[split_name]["train"]["y"],
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": min(target.value_counts().values), # 46
        "n_jobs": 4,
        "return_times": True,
    }
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        mlp_final,
        shuffle=True,
        random_state=0,
        **params,
    )
    # Generate Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 4), sharey=False, sharex=True)

    ax[0].plot(train_sizes, fit_times.mean(axis=1), 'o-')
    ax[0].fill_between(
            train_sizes,
            fit_times.mean(axis=1) - fit_times.std(axis=1),
            fit_times.mean(axis=1) + fit_times.std(axis=1),
            alpha=0.3,
        )
    ax[0].set_ylabel('Fit time (s)')
    ax[0].set_xlabel('# Training Samples')
    ax[0].set_title(
        f"Scalability"
    )
    
    ax[1].plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r', label='Training Score')
    ax[1].plot(train_sizes, test_scores.mean(axis=1), 'o-', color='g', label='CV Score')
    ax[1].fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        color='r',
        alpha=0.3,
    )
    ax[1].fill_between(
        train_sizes,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        color='g',
        alpha=0.3,
    )
    ax[1].set_ylabel('Score')
    ax[1].set_xlabel('# Training Samples')
    ax[1].set_title(
        f"Learning Curve"
    )
    return fig


def generate_confusion_matrices(test_predictions, test_groundtruth):
    EMOTION_MAP = {
        'W': 'wut',        # anger
        'L': 'langeweile', # boredom
        'E': 'ekel',       # disgust
        'A': 'angst',      # fear
        'F': 'freude',     # happiness/joy
        'T': 'trauer',     # sadness
        'N': 'neutral',    # neutral
    }
    cmat = confusion_matrix(test_groundtruth, test_predictions)
    cmat_norm = confusion_matrix(test_groundtruth, test_predictions, normalize='true')

    df_cmat = pd.DataFrame(cmat, index=list(EMOTION_MAP.keys()), columns=list(EMOTION_MAP.keys()))
    df_cmat_norm = pd.DataFrame(cmat_norm, index=list(EMOTION_MAP.keys()), columns=list(EMOTION_MAP.keys()))

    return df_cmat, df_cmat_norm


def generate_confusion_plots(df_cmat: pd.DataFrame, df_cmat_norm: pd.DataFrame):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.title('Confusion Matrix')
    sn.heatmap(df_cmat, annot=True, annot_kws={'size': 18})
    
    plt.subplot(1, 2, 2)
    plt.title('Normalized Confusion Matrix')
    sn.heatmap(df_cmat_norm, annot=True, annot_kws={'size': 18})

    return plt.gcf() 
    

def main(args):
    # Must be set for mlflow to authenticate
    os.environ['AZURE_CLIENT_ID'] = args.app_id
    os.environ['AZURE_CLIENT_SECRET'] = args.client_secret
    os.environ['AZURE_TENANT_ID'] = args.tenant
    # Set MLflow tracking URI
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name,
    )
    mlflow_tracking_uri = ml_client.workspaces.get(name=ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # Create and set experiment
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    print(f"Experiment: {experiment}")
    if experiment is None:
        mlflow.create_experiment(
            name=args.experiment_name,
            tags={
                "pii": False,
                "dataset": "emodb",
                "type": "classifier",
            }
        )
        experiment = mlflow.set_experiment(name=args.experiment_name)
    # Load the Parquet file into a Pandas DataFrame
    X = pd.read_parquet(path=args.parquet_file)
    # Generate emotion distribution plot
    fig1 = plot_emotion_distribution(X)
    # Save `emotion` column and drop from `X`
    y = X["emotion"]
    X.drop(columns=["emotion"], inplace=True)
    # Scale and split
    splits = {
        "standard": split2dict(*split(standard_scale(X), y)),
        "minmax": split2dict(*split(minmax_scale(X), y)),
    }
    # Train and evaluate model
    nn_models = [
        MLPClassifier(random_state=0),
    ]
    nnet_scores = train(nn_models, splits)
    print(nnet_scores)
    # Hyperparameter tuning
    gs = grid_search(splits, "standard")
    print(gs.best_params_)
    # Retrain and evaluate final model
    mlp_final = MLPClassifier(
        **gs.best_params_,
        # Adam constants -- defaults suggested in paper
        beta_1=0.9,
        beta_2=0.999,
        batch_size=32,
        max_iter=200,
        random_state=0,
    )
    mlp_final.fit(splits["standard"]["train"]["X"], splits["standard"]["train"]["y"])
    mlp_final_scores = score(mlp_final, splits, "standard")
    print(mlp_final_scores)
    # StratifiedKFold
    kfold_scores = stratifiedkfold(mlp_final, y, splits, "standard")
    kfold_stats = {
        "kfold_mean": np.mean(kfold_scores),
        "kfold_std": np.std(kfold_scores),
    }
    print(kfold_stats)
    # Generate learning and scaling curves
    fig2 = generate_learning_scaling_curves(y, mlp_final, splits, "standard")
    # Generate confusion matrices
    fig3 = generate_confusion_plots(
        *generate_confusion_matrices(
            mlp_final.predict(splits["standard"]["test"]["X"]), 
            splits["standard"]["test"]["y"]),
    )

    artifact_path = "rt-emodb"
    with mlflow.start_run() as run:
        # Log visualizations
        mlflow.log_figure(fig1, "emotion_distribution.png")
        mlflow.log_figure(fig2, "learning_scaling_curves.png")
        mlflow.log_figure(fig3, "confusion.png")
        # Log GridSearchCV parameters and metrics
        mlflow.log_metric("best_cross_val_score", gs.best_score_)
        mlflow.log_metric("num_params_tested", len(gs.cv_results_["params"]))
        mlflow.log_metric("mean_fit_time", gs.cv_results_["mean_fit_time"][gs.best_index_])
        for param, value in gs.best_params_.items():
            mlflow.log_param(f"grid_search_{param}", value)
        # Log model metrics
        for metric_name, metric_value in mlp_final_scores.items():
            mlflow.log_metric(metric_name, metric_value)
        # Log kfold params and metrics
        mlflow.log_param("kfold_n_splits", min(y.value_counts().values))
        for name, value in kfold_stats.items():
            mlflow.log_metric(name, value)
        # Log model
        mlflow.sklearn.log_model(
            mlp_final,
            artifact_path,
            registered_model_name=args.model_name,
            signature=infer_signature(
                splits["standard"]["test"]["X"],
                mlp_final.predict(splits["standard"]["test"]["X"]),
            ),
            # omit `conda_env`: [mlflow.models.infer_pip_requirements](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.infer_pip_requirements)
        )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main(parse_args())