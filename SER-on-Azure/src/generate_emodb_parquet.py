import argparse
import os

import azure.core
import pandas as pd
import numpy as np
import soundfile
import librosa
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azureml.core import Dataset, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

# DATA MAPS
EMOTION_MAP = {
    'W': 'wut',        # anger
    'L': 'langeweile', # boredom
    'E': 'ekel',       # disgust
    'A': 'angst',      # fear
    'F': 'freude',     # happiness/joy
    'T': 'trauer',     # sadness
    'N': 'neutral',    # neutral
}

SPEAKER_MAP = {
    '03': {
        'gender': 0,
        'age': 31,
    },
    '08': {
        'gender': 1,
        'age': 34,
    },
    '09': {
        'gender': 1,
        'age': 21,
    },
    '10': {
        'gender': 0,
        'age': 32,
    },
    '11': {
        'gender': 0,
        'age': 26,
    },
    '12': {
        'gender': 0,
        'age': 30,
    },
    '13': {
        'gender': 1,
        'age': 32,
    },
    '14': {
        'gender': 1,
        'age': 35,
    },
    '15': {
        'gender': 0,
        'age': 25,
    },
    '16': {
        'gender': 1,
        'age': 31,
    },
}

TEXT_MAP = {
    'a01': {
        'text_de': 'Der Lappen liegt auf dem Eisschrank.',
        'text_en': 'The tablecloth is lying on the fridge.'
    },
    'a02': {
        'text_de': 'Das will sie am Mittwoch abgeben.',
        'text_en': 'She will hand it in on Wednesday.'
    },
    'a04': {
        'text_de': 'Heute abend könnte ich es ihm sagen.',
        'text_en': 'Tonight I could tell him.'
    },
    'a05': {
        'text_de': 'Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.',
        'text_en': 'The black sheet of paper is located up there besides the piece of timber.'
    },
    'a07': {
        'text_de': 'In sieben Stunden wird es soweit sein.',
        'text_en': 'In seven hours it will be.'
    },
    'b01': {
        'text_de': 'Was sind denn das für Tüten, die da unter dem Tisch stehen?',
        'text_en': 'What about the bags standing there under the table?'
    },
    'b02': {
        'text_de': 'Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.',
        'text_en': 'They just carried it upstairs and now they are going down again.'
    },
    'b03': {
        'text_de': 'An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.',
        'text_en': 'Currently at the weekends I always went home and saw Agnes.'
    },
    'b09': {
        'text_de': 'Ich will das eben wegbringen und dann mit Karl was trinken gehen.',
        'text_en': 'I will just discard this and then go for a drink with Karl.'
    },
    'b10': {
        'text_de': 'Die wird auf dem Platz sein, wo wir sie immer hinlegen.',
        'text_en': 'It will be in the place where we always store it.'
    }
}

# PARSING FUNCTIONS
def emotion_of(filename: str) -> str:
    return EMOTION_MAP[filename[-2]]

def parse_filename(filepath: str, dir_path: str) -> dict[str, str]:
    filename = filepath.split('.')[0]
    return {
        **SPEAKER_MAP[filename[:2]],
        **TEXT_MAP[filename[2:5]],
        'filepath': f'{dir_path}/{filepath}',
        'filename': filename,
        'emotion': EMOTION_MAP[filename[-2]],
        'instance': filename[-1],
    }

def map_filenames(dir_path: str) -> dict[str, dict[str, str]]:
    return [parse_filename(filename, dir_path) for filename in os.listdir(dir_path)]

def f_chromagram(waveform, sample_rate):
    """Generate the chromagram of `waveform`'s STFT. Produces 12 features."""
    return np.mean(librosa.feature.chroma_stft(
        S=np.abs(librosa.stft(waveform)),
        sr=sample_rate,
    ).T, axis=0)

def f_mel_spectrogram(waveform, sample_rate):
    """Generate Mel Spectrogram of `waveform`. Generates 128 features."""
    return np.mean(librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
    ).T, axis=0)

def f_mfcc(waveform, sample_rate, n_mfcc: int = 40):
    """Generate `n_mfcc` Mel-Frequency Cepstral Coefficientss of `waveform`. Produces `n_mfcc` features."""
    return np.mean(librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    ).T, axis=0)

def features(filepath):
    with soundfile.SoundFile(filepath) as audio:
        waveform = audio.read(dtype="float32")
        
        feature_matrix = np.array([])
        feature_matrix = np.hstack((
            f_chromagram(waveform, audio.samplerate),
            f_mel_spectrogram(waveform, audio.samplerate),
            f_mfcc(waveform, audio.samplerate),
        ))
    return feature_matrix

# ASSET FUNCTIONS
def bump_patch(version: str) -> str:
    """Given a semver string of form `vx.x.x`, return `v.x.x.{x+1}`."""
    comps = [int(el) for el in version.replace("v", "").split(".")]
    comps[-1] = comps[-1] + 1
    return f"v{'.'.join([str(el) for el in comps])}"

def get_new_asset_version(ml_client: MLClient, asset_name: str) -> str:
    try:
        asset_versions = sorted([asset.version for asset in ml_client.data.list(name=asset_name)])
        return bump_patch(asset_versions[-1])
    except azure.core.exceptions.ResourceNotFoundError:  # If no versions are found
        return "v1.0.0"

def feature_parquet_create_or_update(ml_client: MLClient, parquet_filepath: str, asset_name: str) -> str:
    new_asset = Data(
        name=asset_name,
        version=get_new_asset_version(ml_client, asset_name),
        path=parquet_filepath,
        type=AssetTypes.URI_FILE,
        description="Parquet file describing the EmoDB feature set.",
    )
    return ml_client.data.create_or_update(new_asset)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("-a", "--asset-name", type=str)
    parser.add_argument("-d", "--dataset-name", type=str)
    parser.add_argument("-p", "--parquet-output-filename", type=str)
    parser.add_argument("-o", "--download-path", type=str)
    parser.add_argument("-w", "--workspace-name", type=str)
    parser.add_argument("-s", "--subscription-id", type=str)
    parser.add_argument("-g", "--resource-group-name", type=str)
    parser.add_argument("-i", "--app-id", type=str)
    parser.add_argument("-c", "--client-secret", type=str)
    parser.add_argument("-t", "--tenant", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    ws = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group_name,
        workspace_name=args.workspace_name,
        auth=ServicePrincipalAuthentication(
            tenant_id=args.tenant,
            service_principal_id=args.app_id,
            service_principal_password=args.client_secret,
        )
    )
    
    # Download data
    dataset = Dataset.get_by_name(
        workspace=ws,
        name=args.dataset_name,
        version="latest",
    )
    dataset.download(
        target_path=args.download_path,
        overwrite=True,
    )
    # Parse filenames to extract metadata
    df = pd.DataFrame(map_filenames(args.download_path))

    # Generate features for each audio file
    X = pd.DataFrame(df["filepath"].apply(features).tolist(), index=df.index)
    X["filename"] = df["filename"]
    X.set_index("filename", inplace=True)
    X = X.rename(columns={i: f'feature{i}' for i in range(180)})
    X["emotion"] = df["emotion"].to_numpy()
    print(X.head())

    # Save as Parquet file
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    X.to_parquet(os.path.join(output_dir, args.parquet_output_filename))

    # Upload as URI File
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
    )
    feature_parquet_create_or_update(ml_client, os.path.join(output_dir, args.parquet_output_filename), args.asset_name)


if __name__ == "__main__":
    main(parse_args())