from supabase import create_client, Client
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

def get_supabase_client() -> Client:
    """Create the Supabase client only when authentication is used."""
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be configured for authentication.")
    return create_client(supabase_url, supabase_key)

class Auth:
    def __init__(self, client: Client | None = None):
        self.client = client or get_supabase_client()

    def sign_up(self,email, password):
        try:
            user = self.client.auth.sign_up({"email":email, "password": password})
            return user
        except Exception as e:
            st.error(f"Registration failed {e}")
            return None

    def sign_in(self,email,password):
        try:
            user = self.client.auth.sign_in_with_password({"email":email,"password":password})
            return user
        except Exception as e:
            st.error(f"Failed during sigin {e}")
            return None

    def sign_out(self):
        try:
            self.client.auth.sign_out()
        except Exception as e:
            return e

class AuthScreen:
    def auth_screen(self, auth: Auth | None = None):
        auth = auth or Auth()
        st.title("Authentication Page")
        option = st.selectbox("Choose an action: ",['Login', 'Sign Up'])
        email = st.text_input('Email')
        password = st.text_input("Password",type="password")
        if option == "Login" and st.button('Login'):
            user = auth.sign_in(email,password)
            if user and user.user:
                st.session_state.user_email = user.user.email
                st.success(f"Welcome back, {email}!")
                st.rerun()
        if option == 'Sign Up' and st.button('Register'):
            user = auth.sign_up(email,password)
            if user and user.user:
                st.success("Registration complete. Please Log In")
