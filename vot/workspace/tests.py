

import logging
import tempfile
import unittest
from vot import get_logger
from vot.workspace.storage import Cache, LocalStorage

from vot.workspace import Workspace, NullStorage

class TestStacks(unittest.TestCase):

    def test_void_storage(self):
       
        storage = NullStorage()

        with storage.write("test.data") as handle:
            handle.write("test")

        self.assertIsNone(storage.read("test.data"))

    def test_local_storage(self):
       
        with tempfile.TemporaryDirectory() as testdir:
            storage = LocalStorage(testdir)

            with storage.write("test.txt") as handle:
                handle.write("Test")

            self.assertTrue(storage.isdocument("test.txt"))

        # TODO: more tests

    def test_workspace_create(self):
       
        get_logger().setLevel(logging.WARN) # Disable progress bar

        default_config = dict(stack="testing", registry=["./trackers.ini"])

        with tempfile.TemporaryDirectory() as testdir:
            Workspace.initialize(testdir, default_config, download=True)
            Workspace.load(testdir)

    def test_cache(self):
        with tempfile.TemporaryDirectory() as testdir:

            cache = Cache(LocalStorage(testdir))

            self.assertFalse("test" in cache)

            cache["test"] = 1

            self.assertTrue("test" in cache)

            self.assertTrue(cache["test"] == 1)

            del cache["test"]

            self.assertRaises(KeyError, lambda: cache["test"])