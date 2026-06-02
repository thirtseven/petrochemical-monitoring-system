import argparse
import yaml
import torch
from pathlib import Path

from src.data.data_loader import PetrochemicalDataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMDetector
from src.models.transformer_model import TransformerDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.train import Trainer
from src.utils.visualization import Visualizer
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Petrochemical Monitoring System')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer', 'autoencoder'], help='Model type')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'], help='Operation mode')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logger('PetrochemicalMonitoring', 'results/logs/training.log')
    logger.info("Starting Petrochemical Monitoring System")

    logger.info("Loading data...")
    data_loader = PetrochemicalDataLoader()
    preprocessor = DataPreprocessor(scaler_type='standard')

    X, y = data_loader.load_tennessee_eastman(fault_id=1, split='train')
    X_scaled = preprocessor.fit_transform(X)

    seq_length = config['data']['sequence_length']
    X_seq, y_seq = preprocessor.create_sequences(X_scaled, y, seq_length)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(
        X_seq, y_seq, 
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )

    logger.info(f"Creating {args.model} model...")
    if args.model == 'lstm':
        model = LSTMDetector(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    elif args.model == 'transformer':
        model = TransformerDetector(
            input_dim=config['model']['input_dim'],
            d_model=64,
            nhead=4,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    elif args.model == 'autoencoder':
        model = AutoencoderDetector(
            input_dim=config['model']['input_dim'],
            encoding_dim=32,
            dropout=config['model']['dropout']
        )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.mode == 'train':
        logger.info("Starting training...")
        trainer = Trainer(model, config)
        history = trainer.train(train_loader, val_loader)
        logger.info(f"Training complete. Best validation loss: {history['best_val_loss']:.4f}")

        visualizer = Visualizer()
        visualizer.plot_training_history(
            history['train_losses'],
            history['val_losses'],
            save_path='results/figures/training_history.png'
        )

    elif args.mode == 'evaluate':
        logger.info("Evaluating model...")
        trainer = Trainer(model, config)
        trainer.load_checkpoint('best_model.pth')

        predictions, labels = trainer.predict(test_loader)

        from src.utils.metrics import calculate_anomaly_threshold
        threshold = calculate_anomaly_threshold(predictions, config['anomaly']['threshold_percentile'])
        pred_labels = (predictions > threshold).astype(int)

        metrics = calculate_metrics(labels, pred_labels, predictions)
        logger.info(f"Evaluation metrics: {metrics}")

        visualizer = Visualizer()
        visualizer.plot_anomaly_scores(
            predictions, labels, threshold,
            save_path='results/figures/anomaly_scores.png'
        )
        visualizer.plot_confusion_matrix(
            labels, pred_labels,
            save_path='results/figures/confusion_matrix.png'
        )
        visualizer.plot_roc_curve(
            labels, predictions,
            save_path='results/figures/roc_curve.png'
        )

    logger.info("Done!")

if __name__ == '__main__':
    main()
