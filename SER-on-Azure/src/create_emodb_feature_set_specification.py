import argparse
import logging
import os

import azure.core
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml.entities import Data, DataColumn, DataColumnType, FeatureSet, FeatureSetSpecification, FeatureStoreEntity
from azure.identity import DefaultAzureCredential
from azureml.core import Run
from azureml.featurestore import FeatureSetSpec
from azureml.featurestore.contracts.feature import Feature
from azureml.featurestore.contracts.feature_source_type import SourceType
from azureml.featurestore.contracts.feature_source import FeatureSource
from azureml.featurestore.contracts import (
    Column,
    ColumnType,
    TimestampColumn,
)
from azureml.featurestore.feature_source import ParquetFeatureSource
import pandas as pd
import pyspark


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("-p", "--parquet-output-filename", type=str)
    parser.add_argument("-v", "--version", type=str)
    parser.add_argument("-w", "--workspace-name", type=str)
    parser.add_argument("-s", "--subscription-id", type=str)
    parser.add_argument("-g", "--resource-group-name", type=str)
    parser.add_argument("-q", "--parquet-file", type=str)
    parser.add_argument("-f", "--feature-store-name", type=str)
    parser.add_argument("-i", "--app-id", type=str)
    parser.add_argument("-c", "--client-secret", type=str)
    parser.add_argument("-t", "--tenant", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def bump_patch(version: str) -> str:
    """Given a semver string of form `vx.x.x`, return `v.x.x.{x+1}`."""
    comps = [int(el) for el in version.replace("v", "").split(".")]
    comps[-1] = comps[-1] + 1
    return f"v{'.'.join([str(el) for el in comps])}"


def get_new_feature_set_version(fs_client: MLClient, asset_name: str) -> str:
    try:
        asset_versions = [asset.version for asset in fs_client.feature_sets.list(name=asset_name)]
        if not len(asset_versions):
            # package is just broken, as usual, try to fetch specific versions...how stupid
            latest = 0
            while True:
                version = f"v1.0.{latest}"
                try:
                    fs_client.feature_sets.get(name=asset_name, version=version)
                    latest += 1
                except azure.core.exceptions.ResourceNotFoundError as e:
                    return version
    except azure.core.exceptions.ResourceNotFoundError:
        return "v1.0.0"


def get_new_asset_version(ml_client: MLClient, asset_name: str) -> str:
    try:
        asset_versions = [asset.version for asset in ml_client.data.list(name=asset_name)]
        asset_versions = sorted(asset_versions)
        return bump_patch(asset_versions[-1])
    except azure.core.exceptions.ResourceNotFoundError:  # If no versions are found
        return "v1.0.0"
    

def main(args):
    os.environ['AZURE_CLIENT_ID'] = args.app_id
    os.environ['AZURE_CLIENT_SECRET'] = args.client_secret
    os.environ['AZURE_TENANT_ID'] = args.tenant
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name,
    )

    fs_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.feature_store_name,
    )
    
    # 0. Fetch, download, load Parquet as DF
    X = pd.read_parquet(path=args.parquet_file)

    # 1. Define FeatureSetSpec instance
    parquet_asset_versions = sorted(list(ml_client.data.list(name=args.parquet_output_filename)), key=lambda a: a.version)
    exclusions = ["timestamp", "filename", "filepath", "emotions"]
    fss = FeatureSetSpec(
        source=FeatureSource(
            type=SourceType.PARQUET,
            path=parquet_asset_versions[-1].path,
            timestamp_column=TimestampColumn(name="timestamp"),
        ),
        features=[Feature(name=str(f), type=ColumnType.FLOAT) for f in X.columns.to_list() if f not in exclusions],
        index_columns=[Column(name="filename", type=ColumnType.STRING)],
    )

    # 2. `dump` the YAML -- for some reason, `dump` ignores custom filenames, true to form
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    # using ONLY `output_dir` forces use of `FeatureSetSpec.yaml` as default name
    fss.dump(output_dir, overwrite=True)

    # 3. Create a URI FIle using the result
    fss_yaml_path = "FeatureSetSpec.yaml"
    outpath = os.path.join(output_dir, fss_yaml_path)
    feature_set_specification_yaml_name = "EmoDB-FeatureSetSpecfication"
    feature_set_specification_yaml_asset = Data(
        name=feature_set_specification_yaml_name,
        version=get_new_asset_version(fs_client, feature_set_specification_yaml_name),
        path=outpath,
        type=AssetTypes.URI_FILE,
        description="Feature Set Specification YAML for EmoDB data.",
    )
    try:
        feature_set_specification_yaml_result = ml_client.data.create_or_update(feature_set_specification_yaml_asset)
    except azure.core.exceptions.HttpResponseError as e:
        logging.error(e)
        logging.info(f"Attempted to use: {get_new_asset_version(ml_client, feature_set_specification_yaml_name)}")

    # 4a. Create a Feature Set Entity
    entity_name = "filename"
    entity_versions = sorted([int(e.version) for e in fs_client.feature_store_entities.list(name=entity_name)])
    next_version = str(entity_versions[-1] + 1) if len(entity_versions) > 0 else "1"
    fs_entity = FeatureStoreEntity(
        name=entity_name,
        version=next_version,
        index_columns=[DataColumn(name=entity_name, type=DataColumnType.STRING)],
        stage="Development",
        description=f"This entity represents the index column of the EmoDB dataset, `{entity_name}`.",
        tags={
            "pii": False
        },
    )
    entity_poller = fs_client.feature_store_entities.begin_create_or_update(fs_entity)
    entity = entity_poller.result()

    # 4b. Create Feature Set Object w/ Entity
    feature_set_name = "EmoDB-FeatureSet2"
    feature_set_version = get_new_feature_set_version(fs_client, feature_set_name)
    print(f"Feature Set Version: {feature_set_version}")
    fs_poller = fs_client.feature_sets.begin_create_or_update(featureset=FeatureSet(
        name=feature_set_name,
        version=feature_set_version,
        description="Data Set for training German SER Classifier.",
        entities=[f"azureml:{entity_name}:{next_version}"],
        specification=FeatureSetSpecification(path=output_dir),
        tags={
            "pii": False,
        },
    ))
    fs_result = fs_poller.result()
    logging.info(fs_result)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main(parse_args())