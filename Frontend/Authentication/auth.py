from supabase import create_client, Client
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url,supabase_key) 

class Auth:

    def sign_up(self,email, password):
        try:
            user = supabase.auth.sign_up({"email":email, "password": password})
            return user
        except Exception as e:
            # st.error(f"Registration failed {e}")
            return e

    def sign_in(self,email,password):
        try:
            user = supabase.auth.sign_in_with_password({"email":email,"password":password})
            return user
        except Exception as e:
            # st.error(f"Failed during sigin {e}")
            return e

    def sign_out(self):
        try:
            supabase.auth.sign_out()
        except Exception as e:
            return e