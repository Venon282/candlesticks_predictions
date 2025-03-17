
import os
import sys
import joblib

from functools import partial
from pathlib import PurePath, Path
import numpy as np
import tensorflow as tf

# internal
from py_libraries.ml.optimisation import TransformerModelBuilder
from py_libraries.ml.optimisation import BayesianTuner
from model import DirectionalAccuracy, CustomCandleLoss

"""
Reference: https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=fr
"""

if __name__ == '__main__':
    # global vars
    result_path = Path('./opti')
    datas_path = Path('./datas/split')
    datas_caracteristiques_str = '30_5_None_rsi_macd_bollinger'
    datas_split_str = '9_1_0'
    project_name = datas_caracteristiques_str
    
    # Load datas
    inputs_train    = joblib.load(datas_path / datas_caracteristiques_str / datas_split_str / 'inputs_train.pkl')
    inputs_val      = joblib.load(datas_path / datas_caracteristiques_str / datas_split_str / 'inputs_val.pkl')

    outputs_train   = joblib.load(datas_path / datas_caracteristiques_str / datas_split_str / 'outputs_train.pkl')
    outputs_val     = joblib.load(datas_path / datas_caracteristiques_str / datas_split_str / 'outputs_val.pkl')
    
    # Define inputs outputs
    datas_caracteristiques_split = datas_caracteristiques_str.split('_')
    n_features = inputs_train.shape[-1]
    n_source = int(datas_caracteristiques_split[0])
    n_target = int(datas_caracteristiques_split[1])
    loss = CustomCandleLoss(penalty_direction_weight=2.0)
    metrics = [
                'mse',
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                DirectionalAccuracy(name='directional_accuracy')
              ]
    
    print(f'{n_features=}')
    print(f'{n_source=}')
    print(f'{n_target=}')
    
    print(f'{inputs_train.shape=}')
    print(f'{inputs_val.shape=}')
    print(f'{outputs_train.shape=}')
    print(f'{outputs_val.shape=}')
    
    transformer_model_builder = TransformerModelBuilder(n_features, n_source, n_target, fixe_hparams={}, loss=loss, metrics=metrics)
    transformer_builder = transformer_model_builder.build_model

    # Initialisation du tuner Hyperband
    tuner = BayesianTuner(
        transformer_builder,
        objective='val_loss',      
        max_trials=100, 
        directory=str(result_path),  # Dossier pour sauvegarder les résultats
        project_name=project_name,       # Nom du projet
        fixe_hparams={'batch_size':256, 'early_stopping_patience':40},
        rlrop=False
    )
    
    # Creating a start token for the decoder
    start_token = tf.zeros((n_features,), dtype=tf.float32)

    # Construction of the decoder input for training:
    # For each sequence of the target, we prefix with a start token and we shift (we remove the last element)
    decoder_input_train = tf.concat([
        tf.broadcast_to(tf.expand_dims(start_token, 0), [tf.shape(outputs_train)[0], 1, n_features]),
        outputs_train[:, :-1, :]
    ], axis=1)

    # Construction of the decoder input for validation
    decoder_input_val = tf.concat([
        tf.broadcast_to(tf.expand_dims(start_token, 0), [tf.shape(outputs_val)[0], 1, n_features]),
        outputs_val[:, :-1, :]
    ], axis=1)

    # Lancer la recherche d'hyperparamètres
    tuner.search(
        (inputs_train, decoder_input_train),
        validation_data=((inputs_val, decoder_input_val), outputs_val),
        epochs=200,
        verbose=1
    )

    # Récupération des meilleurs hyperparamètres trouvés
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Reconstruction et entraînement du modèle avec les meilleurs hyperparamètres
    best_model = tuner.hypermodel.build(best_hp)

    # Sauvegarde du meilleur modèle entraîné
    best_model.save(result_path / f'best_model_{project_name}.keras')

    print("Meilleurs hyperparamètres trouvés :")
    for key, value in best_hp.values.items():
        print(f'{key} : {value}')