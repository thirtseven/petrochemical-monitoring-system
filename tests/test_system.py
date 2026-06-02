import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data.data_loader import PetrochemicalDataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMDetector
from src.models.transformer_model import TransformerDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.utils.metrics import calculate_metrics, calculate_anomaly_threshold

def test_data_loader():
    print("Testing Data Loader...")
    loader = PetrochemicalDataLoader()
    X, y = loader.load_tennessee_eastman(fault_id=0, split='train')
    
    assert X.shape[0] > 0, "Data should not be empty"
    assert X.shape[1] == 52, f"Expected 52 features, got {X.shape[1]}"
    assert len(y) == len(X), "Labels length should match data length"
    print(f"  ✓ Data shape: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")
    return True

def test_preprocessor():
    print("\nTesting Preprocessor...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    X = np.random.randn(100, 52)
    y = np.zeros(100)
    
    X_scaled = preprocessor.fit_transform(X)
    assert X_scaled.shape == X.shape, "Scaled shape should match original"
    assert abs(np.mean(X_scaled)) < 0.1, "Scaled data should have near-zero mean"
    assert abs(np.std(X_scaled) - 1.0) < 0.1, "Scaled data should have unit variance"
    
    X_seq, y_seq = preprocessor.create_sequences(X_scaled, y, seq_length=10)
    assert len(X_seq) == len(X_scaled) - 9, "Sequence count should be correct"
    assert X_seq.shape[1] == 10, "Sequence length should be 10"
    
    print(f"  ✓ Scaling works correctly")
    print(f"  ✓ Sequences shape: {X_seq.shape}")
    return True

def test_lstm_model():
    print("\nTesting LSTM Model...")
    model = LSTMDetector(input_dim=52, hidden_dim=64, num_layers=2)
    
    batch = torch.randn(4, 50, 52)
    output = model(batch)
    
    assert output.shape == (4,), f"Expected shape (4,), got {output.shape}"
    assert all(0 <= val <= 1 for val in output.detach().numpy()), "Output should be between 0 and 1"
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Parameters: {param_count:,}")
    return True

def test_transformer_model():
    print("\nTesting Transformer Model...")
    model = TransformerDetector(input_dim=52, d_model=64, nhead=4, num_layers=2)
    
    batch = torch.randn(4, 50, 52)
    output = model(batch)
    
    assert output.shape == (4,), f"Expected shape (4,), got {output.shape}"
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Parameters: {param_count:,}")
    return True

def test_autoencoder_model():
    print("\nTesting Autoencoder Model...")
    model = AutoencoderDetector(input_dim=52, encoding_dim=16)
    
    batch = torch.randn(4, 52)
    reconstructed = model(batch)
    
    assert reconstructed.shape == batch.shape, "Reconstructed shape should match input"
    
    anomaly_score = model.get_anomaly_score(batch)
    assert anomaly_score.shape == (4,), f"Expected shape (4,), got {anomaly_score.shape}"
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Reconstruction shape: {reconstructed.shape}")
    print(f"  ✓ Anomaly score shape: {anomaly_score.shape}")
    print(f"  ✓ Parameters: {param_count:,}")
    return True

def test_metrics():
    print("\nTesting Metrics...")
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.85, 0.15, 0.95])
    
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    assert 'accuracy' in metrics, "Should have accuracy"
    assert 'precision' in metrics, "Should have precision"
    assert 'recall' in metrics, "Should have recall"
    assert 'f1' in metrics, "Should have f1"
    assert 'auc_roc' in metrics, "Should have auc_roc"
    
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= metrics['auc_roc'] <= 1, "AUC-ROC should be between 0 and 1"
    
    threshold = calculate_anomaly_threshold(y_scores, percentile=80)
    assert 0 <= threshold <= 1, "Threshold should be between 0 and 1"
    
    print(f"  ✓ Accuracy: {metrics['accuracy']:.3f}")
    print(f"  ✓ Precision: {metrics['precision']:.3f}")
    print(f"  ✓ Recall: {metrics['recall']:.3f}")
    print(f"  ✓ F1: {metrics['f1']:.3f}")
    print(f"  ✓ AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  ✓ Threshold (80th percentile): {threshold:.3f}")
    return True

def test_training_pipeline():
    print("\nTesting Training Pipeline...")
    from src.train import Trainer
    
    model = LSTMDetector(input_dim=52, hidden_dim=32, num_layers=1)
    
    config = {
        'training': {
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'epochs': 3,
            'patience': 2
        },
        'output': {
            'models_path': 'results/models/'
        }
    }
    
    trainer = Trainer(model, config, device='cpu')
    
    X_train = torch.randn(20, 50, 52)
    y_train = torch.randint(0, 2, (20,)).float()
    X_val = torch.randn(5, 50, 52)
    y_val = torch.randint(0, 2, (5,)).float()
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    history = trainer.train(train_loader, val_loader, epochs=3)
    
    assert 'train_losses' in history, "Should have train_losses"
    assert 'val_losses' in history, "Should have val_losses"
    assert len(history['train_losses']) == 3, "Should have 3 epochs"
    
    print(f"  ✓ Training completed: {len(history['train_losses'])} epochs")
    print(f"  ✓ Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"  ✓ Final val loss: {history['val_losses'][-1]:.4f}")
    return True

def run_all_tests():
    print("=" * 60)
    print("Running Petrochemical Monitoring System Tests")
    print("=" * 60)
    
    tests = [
        test_data_loader,
        test_preprocessor,
        test_lstm_model,
        test_transformer_model,
        test_autoencoder_model,
        test_metrics,
        test_training_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
