import unittest

from ber.normalization import (
    contains_non_latin,
    normalize_basic,
    normalize_punctuation,
    number_jaccard,
    probable_postal_token,
    remove_legal_suffixes,
    script_signature,
    token_jaccard,
)


class NormalizationTests(unittest.TestCase):
    def test_unicode_compatibility_normalization_and_casefold(self):
        self.assertEqual(normalize_basic(" ＡCME  LTD. "), "acme ltd.")
        self.assertEqual(normalize_basic("e\u0301"), normalize_basic("é"))

    def test_punctuation_becomes_boundaries(self):
        self.assertEqual(normalize_punctuation("A.B-C"), "a b c")

    def test_multilingual_script_detection_preserves_text(self):
        value = "Lakshmi लक्ष्मी"
        self.assertEqual(script_signature(value), "DEVANAGARI|LATIN")
        self.assertTrue(contains_non_latin(value))
        self.assertEqual(normalize_basic("लक्ष्मी"), "लक्ष्मी")

    def test_address_and_token_features(self):
        self.assertEqual(probable_postal_token("Agra 282005"), "282005")
        self.assertEqual(number_jaccard("14 Main Road 282005", "14 MG Rd 282005"), 1.0)
        self.assertGreater(token_jaccard("Acme Private Limited", "Acme Pvt Ltd"), 0)
        self.assertEqual(remove_legal_suffixes("Acme Private Limited"), "acme")
        self.assertEqual(remove_legal_suffixes("Trust Bank"), "trust bank")
        self.assertEqual(remove_legal_suffixes("Foundation Coffee Company"), "foundation coffee")


if __name__ == "__main__":
    unittest.main()
