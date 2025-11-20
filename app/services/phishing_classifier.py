"""
Phishing Classifier Service

This service loads and manages the phishing classification model,
handling feature extraction and prediction.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import joblib
import pandas as pd
import numpy as np

# Add phishing-classifier to path for feature extraction
phishing_classifier_path = Path(__file__).parent.parent.parent.parent / "phishing-classifier"
if phishing_classifier_path.exists():
    sys.path.insert(0, str(phishing_classifier_path))
    try:
        from src.features.pipeline import create_feature_pipeline
        FEATURE_PIPELINE_AVAILABLE = True
    except ImportError:
        FEATURE_PIPELINE_AVAILABLE = False
        logging.warning("Could not import feature pipeline from phishing-classifier")
else:
    FEATURE_PIPELINE_AVAILABLE = False
    logging.warning("phishing-classifier path not found")

logger = logging.getLogger(__name__)


class PhishingClassifierService:
    """Service for loading and using the phishing classification model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the phishing classifier service.
        
        Args:
            model_path: Path to the model file. If None, uses default location.
        """
        self.model = None
        self.feature_pipeline = None
        self.model_path = model_path or Path(__file__).parent.parent / "models" / "model.joblib"
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found at {self.model_path}")
                return False
                
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            # Initialize feature pipeline
            if FEATURE_PIPELINE_AVAILABLE:
                try:
                    self.feature_pipeline = create_feature_pipeline()
                    # Fit the pipeline with a dummy URL to initialize it
                    dummy_df = pd.DataFrame({'url': ['https://example.com']})
                    self.feature_pipeline.fit(dummy_df)
                    logger.info("Feature pipeline initialized successfully")
                except Exception as e:
                    logger.warning(f"Could not initialize feature pipeline: {e}")
                    self.feature_pipeline = None
            else:
                logger.warning("Feature pipeline not available - using fallback feature extraction")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def extract_features(self, url: str) -> Optional[np.ndarray]:
        """
        Extract features from a URL.
        
        Args:
            url: URL string to extract features from
            
        Returns:
            Feature array or None if extraction fails
        """
        if self.feature_pipeline:
            try:
                # Use the feature pipeline
                df = pd.DataFrame({'url': [url]})
                features_df = self.feature_pipeline.transform(df)
                feature_names = self.feature_pipeline.get_feature_names()
                features = features_df[feature_names].values[0]
                return features
            except Exception as e:
                logger.error(f"Error extracting features with pipeline: {e}")
                return None
        else:
            # Fallback: simple feature extraction
            return self._extract_features_fallback(url)
    
    def _extract_features_fallback(self, url: str) -> np.ndarray:
        """
        Fallback feature extraction when pipeline is not available.
        This is a simplified version that extracts basic features.
        
        Args:
            url: URL string
            
        Returns:
            Feature array (20 features to match model expectations)
        """
        import re
        
        url_lower = url.lower()
        
        # URL features (6)
        url_length = len(url)
        subdomain_count = url.count('.') - 1 if '.' in url else 0
        suspicious_chars = sum(1 for c in url if c in '@#$%&*')
        is_shortened = any(short in url_lower for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co'])
        has_ip = bool(re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url))
        redirect_count = url.count('//') - 1 if '//' in url else 0
        
        # Domain features (6) - simplified
        domain_length = len(url.split('/')[2]) if '/' in url else len(url)
        domain_age_days = 365  # Default
        registrar_score = 0.5  # Default
        country_score = 0.5  # Default
        alexa_score = 0.5  # Default
        ssl_score = 1.0 if url.startswith('https') else 0.0
        
        # Content features (8) - simplified
        suspicious_keywords = sum(1 for kw in ['login', 'verify', 'update', 'secure', 'account'] if kw in url_lower)
        form_count = 0  # Would require HTML parsing
        external_link_ratio = 0.5  # Default
        image_ratio = 0.3  # Default
        js_ratio = 0.2  # Default
        load_time_score = 0.5  # Default
        meta_count = 0  # Would require HTML parsing
        title_length = 50  # Default
        
        features = np.array([
            url_length, subdomain_count, suspicious_chars, is_shortened, has_ip, redirect_count,
            domain_age_days, registrar_score, country_score, alexa_score, ssl_score, domain_length,
            suspicious_keywords, form_count, external_link_ratio, image_ratio, js_ratio,
            load_time_score, meta_count, title_length
        ], dtype=np.float32)
        
        return features
    
    def predict(self, url: str) -> Dict[str, Any]:
        """
        Predict if a URL is phishing or legitimate.
        
        Args:
            url: URL string to classify
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.extract_features(url)
        if features is None:
            raise ValueError("Failed to extract features from URL")
        
        # Reshape for single prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(features)[0]
                confidence = float(max(proba))
                # Get the probability of the predicted class
                predicted_class_idx = int(prediction)
                confidence = float(proba[predicted_class_idx])
            except Exception as e:
                logger.warning(f"Could not get prediction probability: {e}")
        
        # Convert prediction to label
        label = "phishing" if prediction == 1 else "legitimate"
        score = confidence if confidence else (0.9 if prediction == 1 else 0.1)
        
        return {
            "prediction": label,
            "confidence": confidence if confidence else score,
            "score": float(score),
            "predicted_class": int(prediction)
        }


# Global service instance
_classifier_service: Optional[PhishingClassifierService] = None


def get_classifier_service() -> PhishingClassifierService:
    """
    Get or create the global classifier service instance.
    
    Returns:
        PhishingClassifierService instance
    """
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = PhishingClassifierService()
        _classifier_service.load_model()
    return _classifier_service


def initialize_classifier_service(model_path: Optional[Path] = None) -> bool:
    """
    Initialize the global classifier service.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        True if initialization successful, False otherwise
    """
    global _classifier_service
    _classifier_service = PhishingClassifierService(model_path)
    return _classifier_service.load_model()

