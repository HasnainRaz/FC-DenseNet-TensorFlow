import unittest
from helpers import get_data_paths_list
from main import main

class TestMain(unittest.TestCase):

    def test_get_data_paths_wrong(self):
        with self.assertRaises(Exception) as context:
            get_data_paths_list('some/unknown/path', 'other/unknown/path')

            self.assertTrue('No such file or directory' in context.exception)
    
    def test_get_data_paths_correct(self):
        images, masks = get_data_paths_list('repo-images', 'repo-images')

        self.assertGreater(len(images), 0)
        self.assertEqual(len(images), len(masks))