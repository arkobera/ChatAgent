import unittest
from unittest.mock import patch

from Frontend.Authentication.auth import Auth


class FakeSupabaseAuth:
    def __init__(self):
        self.sign_up_calls = []
        self.sign_in_calls = []
        self.did_sign_out = False
        self.sign_up_result = object()
        self.sign_in_result = object()

    def sign_up(self, credentials):
        self.sign_up_calls.append(credentials)
        return self.sign_up_result

    def sign_in_with_password(self, credentials):
        self.sign_in_calls.append(credentials)
        return self.sign_in_result

    def sign_out(self):
        self.did_sign_out = True


class FakeClient:
    def __init__(self):
        self.auth = FakeSupabaseAuth()


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.auth = Auth(self.client)

    def test_sign_up_passes_credentials_to_supabase(self):
        result = self.auth.sign_up("user@example.com", "secure-password")

        self.assertIs(result, self.client.auth.sign_up_result)
        self.assertEqual(
            self.client.auth.sign_up_calls,
            [{"email": "user@example.com", "password": "secure-password"}],
        )

    def test_sign_in_passes_credentials_to_supabase(self):
        result = self.auth.sign_in("user@example.com", "secure-password")

        self.assertIs(result, self.client.auth.sign_in_result)
        self.assertEqual(
            self.client.auth.sign_in_calls,
            [{"email": "user@example.com", "password": "secure-password"}],
        )

    def test_sign_out_delegates_to_supabase(self):
        self.assertIsNone(self.auth.sign_out())
        self.assertTrue(self.client.auth.did_sign_out)

    def test_failed_sign_in_shows_an_error_and_returns_none(self):
        def fail_sign_in(_credentials):
            raise ValueError("invalid credentials")

        self.client.auth.sign_in_with_password = fail_sign_in
        with patch("Frontend.Authentication.auth.st.error") as error:
            result = self.auth.sign_in("user@example.com", "wrong-password")

        self.assertIsNone(result)
        error.assert_called_once()
        self.assertIn("invalid credentials", error.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
