"""
Unit tests for SentimentAnalyzer class.
Tests sentiment analysis with DistilBERT model and batch processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from src.data.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample news data for testing
        self.sample_news_data = [
            {
                'symbol': 'RELIANCE',
                'title': 'Reliance Industries reports strong Q4 results',
                'summary': 'Company shows growth in all segments',
                'publisher': 'Economic Times',
                'publish_time': 1642204800,  # Jan 15, 2022
                'url': 'https://example.com/news1'
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Reliance stock crashes amid market volatility',
                'summary': 'Major losses reported',
                'publisher': 'Business Standard',
                'publish_time': 1642291200,  # Jan 16, 2022
                'url': 'https://example.com/news2'
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Neutral outlook for Reliance shares',
                'summary': 'Analysts maintain hold rating',
                'publisher': 'Mint',
                'publish_time': 1642377600,  # Jan 17, 2022
                'url': 'https://example.com/news3'
            }
        ]
        
        # Mock prediction results
        self.mock_positive_result = [
            {'label': 'POSITIVE', 'score': 0.8},
            {'label': 'NEGATIVE', 'score': 0.2}
        ]
        
        self.mock_negative_result = [
            {'label': 'POSITIVE', 'score': 0.3},
            {'label': 'NEGATIVE', 'score': 0.7}
        ]
        
        self.mock_neutral_result = [
            {'label': 'POSITIVE', 'score': 0.5},
            {'label': 'NEGATIVE', 'score': 0.5}
        ]
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_init_success(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test successful initialization of SentimentAnalyzer."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Test initialization
        analyzer = SentimentAnalyzer()
        
        # Assertions
        self.assertIsNotNone(analyzer.tokenizer)
        self.assertIsNotNone(analyzer.model)
        self.assertIsNotNone(analyzer.pipeline)
        self.assertEqual(analyzer.batch_size, 32)
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()
    
    @patch('src.data.sentiment_analyzer.torch.cuda.is_available')
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_init_device_selection(self, mock_pipeline, mock_model, mock_tokenizer, mock_cuda):
        """Test device selection during initialization."""
        # Test CUDA available
        mock_cuda.return_value = True
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        self.assertEqual(analyzer.device, "cuda")
        
        # Test CUDA not available
        mock_cuda.return_value = False
        analyzer = SentimentAnalyzer()
        self.assertEqual(analyzer.device, "cpu")
        
        # Test explicit device
        analyzer = SentimentAnalyzer(device="cpu")
        self.assertEqual(analyzer.device, "cpu")
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_sentiment_positive(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment analysis for positive text."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [self.mock_positive_result]
        mock_pipeline.return_value = mock_pipeline_instance
        
        analyzer = SentimentAnalyzer()
        
        # Test positive sentiment
        result = analyzer.analyze_sentiment("This is great news!")
        
        # Assertions
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Should be positive
        self.assertLessEqual(result, 1.0)  # Should be within bounds
        mock_pipeline_instance.assert_called_once()
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_sentiment_negative(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment analysis for negative text."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [self.mock_negative_result]
        mock_pipeline.return_value = mock_pipeline_instance
        
        analyzer = SentimentAnalyzer()
        
        # Test negative sentiment
        result = analyzer.analyze_sentiment("This is terrible news!")
        
        # Assertions
        self.assertIsInstance(result, float)
        self.assertLess(result, 0)  # Should be negative
        self.assertGreaterEqual(result, -1.0)  # Should be within bounds
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_sentiment_empty_text(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment analysis for empty or invalid text."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test empty text
        result = analyzer.analyze_sentiment("")
        self.assertEqual(result, 0.0)
        
        # Test None text
        result = analyzer.analyze_sentiment(None)
        self.assertEqual(result, 0.0)
        
        # Test whitespace only
        result = analyzer.analyze_sentiment("   ")
        self.assertEqual(result, 0.0)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_sentiment_batch(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test batch sentiment analysis."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline_instance = Mock()
        
        # Mock batch results
        mock_pipeline_instance.return_value = [
            self.mock_positive_result,
            self.mock_negative_result,
            self.mock_neutral_result
        ]
        mock_pipeline.return_value = mock_pipeline_instance
        
        analyzer = SentimentAnalyzer(batch_size=2)
        
        # Test batch processing
        texts = [
            "Great news for the company!",
            "Stock market crashed today.",
            "Neutral earnings report."
        ]
        
        results = analyzer.analyze_sentiment_batch(texts)
        
        # Assertions
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(score, float) for score in results))
        self.assertTrue(all(-1.0 <= score <= 1.0 for score in results))
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_sentiment_batch_empty(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test batch sentiment analysis with empty input."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test empty list
        results = analyzer.analyze_sentiment_batch([])
        self.assertEqual(results, [])
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_news_headlines(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment analysis for news headlines."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [
            self.mock_positive_result,
            self.mock_negative_result,
            self.mock_neutral_result
        ]
        mock_pipeline.return_value = mock_pipeline_instance
        
        analyzer = SentimentAnalyzer()
        
        # Test news headline analysis
        results = analyzer.analyze_news_headlines(self.sample_news_data)
        
        # Assertions
        self.assertEqual(len(results), 3)
        for news_item in results:
            self.assertIn('sentiment_score', news_item)
            self.assertIsInstance(news_item['sentiment_score'], float)
            self.assertTrue(-1.0 <= news_item['sentiment_score'] <= 1.0)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_news_headlines_empty(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment analysis for empty news data."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test empty news data
        results = analyzer.analyze_news_headlines([])
        self.assertEqual(results, [])
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_analyze_news_headlines_invalid_titles(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment analysis for news with invalid titles."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test news with invalid titles
        invalid_news = [
            {'title': '', 'summary': 'Empty title'},
            {'title': None, 'summary': 'None title'},
            {'summary': 'No title field'},
            {'title': 'Valid title', 'summary': 'Valid news'}
        ]
        
        results = analyzer.analyze_news_headlines(invalid_news)
        
        # Assertions
        self.assertEqual(len(results), 4)
        # First three should have neutral sentiment (0.0)
        for i in range(3):
            self.assertEqual(results[i]['sentiment_score'], 0.0)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_get_aggregated_sentiment_mean(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test aggregated sentiment calculation with mean method."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test data with sentiment scores
        news_with_sentiment = [
            {'title': 'News 1', 'sentiment_score': 0.8},
            {'title': 'News 2', 'sentiment_score': -0.6},
            {'title': 'News 3', 'sentiment_score': 0.2}
        ]
        
        # Test mean aggregation
        result = analyzer.get_aggregated_sentiment(news_with_sentiment, "mean")
        expected = (0.8 + (-0.6) + 0.2) / 3
        self.assertAlmostEqual(result, expected, places=5)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_get_aggregated_sentiment_median(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test aggregated sentiment calculation with median method."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test data with sentiment scores
        news_with_sentiment = [
            {'title': 'News 1', 'sentiment_score': 0.8},
            {'title': 'News 2', 'sentiment_score': -0.6},
            {'title': 'News 3', 'sentiment_score': 0.2}
        ]
        
        # Test median aggregation
        result = analyzer.get_aggregated_sentiment(news_with_sentiment, "median")
        self.assertEqual(result, 0.2)  # Median of [-0.6, 0.2, 0.8]
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_get_aggregated_sentiment_empty(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test aggregated sentiment calculation with empty data."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test empty data
        result = analyzer.get_aggregated_sentiment([])
        self.assertEqual(result, 0.0)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_clean_text(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test text cleaning functionality."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test various text cleaning scenarios
        self.assertEqual(analyzer._clean_text(""), "")
        self.assertEqual(analyzer._clean_text(None), "")
        self.assertEqual(analyzer._clean_text("  hello world  "), "hello world")
        self.assertEqual(analyzer._clean_text("hello    world"), "hello world")
        
        # Test long text truncation
        long_text = "a" * 600
        cleaned = analyzer._clean_text(long_text)
        self.assertLessEqual(len(cleaned), 512)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_convert_to_sentiment_score(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test conversion of model predictions to sentiment scores."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test positive prediction
        positive_score = analyzer._convert_to_sentiment_score(self.mock_positive_result)
        self.assertGreater(positive_score, 0)
        self.assertLessEqual(positive_score, 1.0)
        
        # Test negative prediction
        negative_score = analyzer._convert_to_sentiment_score(self.mock_negative_result)
        self.assertLess(negative_score, 0)
        self.assertGreaterEqual(negative_score, -1.0)
        
        # Test neutral prediction
        neutral_score = analyzer._convert_to_sentiment_score(self.mock_neutral_result)
        self.assertAlmostEqual(abs(neutral_score), 0.5, places=1)  # Should be close to neutral
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_get_model_info(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test getting model information."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab = {'word1': 1, 'word2': 2}
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test model info
        info = analyzer.get_model_info()
        
        # Assertions
        self.assertIn('model_name', info)
        self.assertIn('device', info)
        self.assertIn('batch_size', info)
        self.assertIn('tokenizer_vocab_size', info)
        self.assertIn('model_loaded', info)
        self.assertIn('pipeline_loaded', info)
        
        self.assertTrue(info['model_loaded'])
        self.assertTrue(info['pipeline_loaded'])
        self.assertEqual(info['tokenizer_vocab_size'], 2)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_get_sentiment_label(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test sentiment score to label conversion."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        analyzer = SentimentAnalyzer()
        
        # Test different score ranges
        self.assertEqual(analyzer._get_sentiment_label(0.5), "Positive")
        self.assertEqual(analyzer._get_sentiment_label(-0.5), "Negative")
        self.assertEqual(analyzer._get_sentiment_label(0.05), "Neutral")
        self.assertEqual(analyzer._get_sentiment_label(-0.05), "Neutral")
        self.assertEqual(analyzer._get_sentiment_label(0.0), "Neutral")
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_test_model(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test model testing functionality."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab = {'word1': 1, 'word2': 2}  # Mock vocab with length
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [self.mock_positive_result]
        mock_pipeline.return_value = mock_pipeline_instance
        
        analyzer = SentimentAnalyzer()
        
        # Test model with default texts
        result = analyzer.test_model()
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertIn('test_results', result)
        self.assertIn('model_info', result)
        self.assertGreater(len(result['test_results']), 0)
        
        # Check test result structure
        for test_result in result['test_results']:
            self.assertIn('text', test_result)
            self.assertIn('sentiment_score', test_result)
            self.assertIn('sentiment_label', test_result)
    
    @patch('src.data.sentiment_analyzer.AutoTokenizer')
    @patch('src.data.sentiment_analyzer.AutoModelForSequenceClassification')
    @patch('src.data.sentiment_analyzer.pipeline')
    def test_error_handling(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test error handling in sentiment analysis."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.side_effect = Exception("Model error")
        mock_pipeline.return_value = mock_pipeline_instance
        
        analyzer = SentimentAnalyzer()
        
        # Test error handling in single analysis
        result = analyzer.analyze_sentiment("Test text")
        self.assertEqual(result, 0.0)  # Should return neutral on error
        
        # Test error handling in batch analysis
        results = analyzer.analyze_sentiment_batch(["Text 1", "Text 2"])
        self.assertEqual(results, [0.0, 0.0])  # Should return neutral scores


if __name__ == '__main__':
    unittest.main()