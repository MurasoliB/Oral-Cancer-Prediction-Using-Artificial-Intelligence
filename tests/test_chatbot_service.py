import unittest
from services.chatbot_service import faq

class ChatbotKnowledgeBaseTests(unittest.TestCase):
    def test_result_uses_supplied_server_context(self):
        answer = faq('What do these results mean?', {'prediction': 'Pre-Cancer', 'confidence': 87})
        self.assertIn('Pre-Cancer', answer)
        self.assertIn('87%', answer)
        self.assertIn('not a diagnosis', answer)

    def test_confidence_is_not_diagnosis(self):
        answer = faq('What does the confidence score mean?')
        self.assertIn('not the probability', answer)

    def test_gradcam(self):
        self.assertIn('heatmap', faq('Explain Grad-CAM').lower())

    def test_prevention(self):
        self.assertIn('tobacco', faq('How can I prevent oral cancer?').lower())

    def test_emergency_takes_priority(self):
        self.assertIn('emergency', faq('I cannot breathe').lower())

    def test_unknown_question_offers_guidance(self):
        self.assertIn('I can help', faq('Tell me something obscure').lower() if False else faq('Tell me something obscure'))

if __name__ == '__main__':
    unittest.main()
