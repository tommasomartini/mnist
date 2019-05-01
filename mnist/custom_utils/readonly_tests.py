import unittest

import mnist.custom_utils.readonly as ro


class TestReadOnly(unittest.TestCase):

    def test_edit_attribute(self):
        expected_A = 10

        class MyClass(ro.ReadOnly):
            A = expected_A

        # Assignment.
        with self.assertRaises(ro.ReadOnlyError):
            MyClass.A = 20

        # Built-in function.
        with self.assertRaises(ro.ReadOnlyError):
            setattr(MyClass, 'A', 20)

        self.assertEqual(MyClass.A, expected_A)

    def test_add_attribute(self):
        class MyClass(ro.ReadOnly):
            A = 10

        # Assignment.
        with self.assertRaises(ro.ReadOnlyError):
            MyClass.B = 20

        # Built-in function.
        with self.assertRaises(ro.ReadOnlyError):
            setattr(MyClass, 'B', 20)

    def test_delete_attribute(self):
        class MyClass(ro.ReadOnly):
            A = 10

        # Assignment.
        with self.assertRaises(ro.ReadOnlyError):
            del MyClass.A

        # Built-in function.
        with self.assertRaises(ro.ReadOnlyError):
            delattr(MyClass, 'A')


if __name__ == '__main__':
    unittest.main()
