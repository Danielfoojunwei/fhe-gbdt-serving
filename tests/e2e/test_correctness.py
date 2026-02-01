import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

class TestCorrectnessMetrics(unittest.TestCase):
    def test_classification_auc(self):
        # Mock plaintext vs encrypted logits
        plaintext_logits = np.array([0.1, 0.4, 0.6, 0.9])
        encrypted_logits = np.array([0.12, 0.38, 0.62, 0.88])
        y_true = np.array([0, 0, 1, 1])
        
        auc_plain = roc_auc_score(y_true, plaintext_logits)
        auc_enc = roc_auc_score(y_true, encrypted_logits)
        
        # AUC should be very close
        self.assertAlmostEqual(auc_plain, auc_enc, delta=0.05)

    def test_regression_mae_rmse(self):
        plaintext_preds = np.array([1.0, 2.0, 3.0, 4.0])
        encrypted_preds = np.array([1.02, 1.98, 3.01, 4.05])
        
        mae = mean_absolute_error(plaintext_preds, encrypted_preds)
        rmse = np.sqrt(mean_squared_error(plaintext_preds, encrypted_preds))
        
        self.assertLess(mae, 0.1)
        self.assertLess(rmse, 0.1)

    def test_rank_correlation(self):
        plaintext_preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        encrypted_preds = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        corr, _ = spearmanr(plaintext_preds, encrypted_preds)
        self.assertGreater(corr, 0.95)

if __name__ == '__main__':
    unittest.main()
