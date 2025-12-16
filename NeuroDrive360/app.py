"""Hugging Face Spaces entrypoint for the Streamlit dashboard."""

from streamlit_app.dashboard import main as run_dashboard


if __name__ == "__main__":
    run_dashboard()

