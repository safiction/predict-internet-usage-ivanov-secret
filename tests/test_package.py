import unittest

import nbformat

from src.package import anonymity_issues, sanitized_notebook


class PackageTests(unittest.TestCase):
    def test_runtime_outputs_and_personal_metadata_removed(self):
        path = "/" + "home" + "/" + "student" + "/" + "project"
        cell = nbformat.v4.new_code_cell("print('hello')", outputs=[nbformat.v4.new_output("stream", name="stdout", text=path)])
        cell.metadata = {"execution": {"path": path}}
        notebook = nbformat.v4.new_notebook(cells=[cell], metadata={"author": "student", "path": path})
        sanitized = sanitized_notebook(notebook)
        self.assertEqual(sanitized.cells[0].source, notebook.cells[0].source)
        self.assertEqual(sanitized.cells[0].outputs, [])
        self.assertEqual(sanitized.cells[0].metadata, {})
        self.assertNotIn("author", sanitized.metadata)
        self.assertEqual(anonymity_issues(nbformat.writes(sanitized)), [])

    def test_actual_user_paths_detected(self):
        path = "/" + "home" + "/" + "student" + "/" + "project"
        self.assertTrue(anonymity_issues(path))
        self.assertFalse(anonymity_issues("data/train.csv; ~/.kaggle/kaggle.json"))


if __name__ == "__main__":
    unittest.main()
