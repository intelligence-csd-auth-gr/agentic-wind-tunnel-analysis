import streamlit as st
import subprocess
import os
import shutil
import sys
import pandas as pd
import plotly.io as pio
from supabase import create_client, Client


current_dir = os.path.dirname(os.path.abspath(__file__))


# --- SUPABASE INITIALIZATION (one client per browser session) ---
def get_supabase() -> Client:
    if "supabase" not in st.session_state:
        st.session_state.supabase = create_client(
            st.secrets["supabase"]["url"],
            st.secrets["supabase"]["key"],  # anon key
        )
    return st.session_state.supabase

supabase = get_supabase()


# --- UI HELPERS ---
def spacer(height=1.0):
    st.markdown(f"<div style='height:{height}rem'></div>", unsafe_allow_html=True)


# --- HISTORY ---
def load_history(limit=20):
    try:
        res = (
            supabase.table("queries")
            .select("id, query_text, selected_tool, created_at")
            .eq("user_id", st.session_state.user_info["id"])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data
    except Exception as e:
        print("Failed to load history:", e)
        return []

def use_query(text):
    st.session_state.query_input = text

def render_history():
    with st.expander("History", expanded=False):
        if not st.session_state.logged_in:
            st.caption("Log in to save and see your query history.")
            return
        rows = load_history()
        if not rows:
            st.caption("No queries yet.")
            return
        for row in rows:
            text = row["query_text"] or ""
            label = text if len(text) <= 40 else text[:40] + "…"
            st.button(
                label,
                key=f"hist_{row['id']}",
                help=f"Tool: {row['selected_tool'] or 'n/a'}\n\n{text}",
                on_click=use_query,
                args=(text,),
                use_container_width=True,
            )


# --- HELP WINDOW ---
EXAMPLE_QUERIES = {
    "Hysteresis": "Compare dynamic stall hysteresis loops for runs 470 and 473.",
    "Cp distribution (Flow separation)": "Show chordwise Cp distribution for runs 470 and 473 at AOA 15.0 deg.",
    "Integrated vs reported": "Compare integrated pressure tap values vs reported force coefficients for run 470.",
    "Center of pressure": "Calculate the center of pressure and generate the stability plot for Run 470.",
    "Aerodynamic efficiency": "Plot the aerodynamic efficiency for Run 470.",
    "Cp between times": "Plot the pressure distribution between 0.5s and 0.75s for runs 470 and 488.",
    "Cp at different AOAs": "Plot pressure distribution for Run 470 at AOA 12.0 deg and 15.0 deg.",
}

@st.dialog("Example queries", width="large")
def show_help():
    st.caption("Pick an example and it will be copied into the Query box.")
    for name, text in EXAMPLE_QUERIES.items():
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.code(text, language=None)
            if st.button("Use this query", key=f"example_{name}"):
                st.session_state.pending_query = text
                st.rerun()  # closes the window and refreshes the page


# --- SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "is_guest" not in st.session_state:
    st.session_state.is_guest = False
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

# ==========================================
# --- LANDING PAGE (AUTHENTICATION VIEW) ---
# ==========================================
if not st.session_state.logged_in and not st.session_state.is_guest:

    st.title("AWTA: Agentic Wind Tunnel Analysis")
    st.markdown("### Welcome! How You Want to Continue?")

    with st.container(border=True):
        st.markdown("**Why log in?**")
        st.markdown(
            "- 📜 **Query history:** see and reuse all your previous queries\n"
            "- 🛠️ **Help us improve:** your queries help us make the app better"
        )
        st.caption("Guests can use every feature, but their history is not saved.")

    spacer()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Login", use_container_width=True):
            st.session_state.auth_mode = "Login"
    with col2:
        if st.button("Register", use_container_width=True):
            st.session_state.auth_mode = "Register"
    with col3:
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.is_guest = True
            st.session_state.auth_mode = None
            st.rerun()

    if st.session_state.auth_mode == "Register":
        st.divider()
        st.subheader("Create a new account")
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            submit_register = st.form_submit_button("Register")

            if submit_register:
                if new_username and new_password and new_email:
                    try:
                        supabase.auth.sign_up({
                            "email": new_email,
                            "password": new_password,
                            "options": {"data": {"username": new_username}},
                        })
                        st.success("Registration successful! Check your email to confirm (if enabled), then log in.")
                    except Exception as e:
                        st.error(f"Registration error: {e}")
                else:
                    st.warning("Please fill all fields.")

    elif st.session_state.auth_mode == "Login":
        st.divider()
        st.subheader("Login to your account")
        with st.form("login_form"):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")

            if submit_login:
                try:
                    res = supabase.auth.sign_in_with_password(
                        {"email": login_email, "password": login_password}
                    )
                    profile = (
                        supabase.table("profiles")
                        .select("username")
                        .eq("id", res.user.id)
                        .maybe_single()
                        .execute()
                    )
                    username = profile.data["username"] if profile and profile.data else login_email
                    st.session_state.logged_in = True
                    st.session_state.user_info = {
                        "id": res.user.id,
                        "email": res.user.email,
                        "username": username,
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
else:
    # ==========================================
    # --- MAIN APPLICATION VIEW ---
    # ==========================================

    # Apply a query chosen in the Help window (must run BEFORE the text area is created)
    if "pending_query" in st.session_state:
        st.session_state.query_input = st.session_state.pop("pending_query")

    # Sidebar layout, top to bottom (containers keep their order even if filled later)
    with st.sidebar:
        st.header("Settings")
        api_box = st.container()
        st.divider()
        files_box = st.container()
        spacer()
        query_box = st.container()
        spacer()
        buttons_box = st.container()
        spacer()
        history_box = st.container()
        spacer(3)
        logout_box = st.container()

    with api_box:
        user_api_key = st.text_input("TypeSafe API Key:", type="password")

    # Logout is filled before st.stop() so it is always available
    with logout_box:
        st.divider()
        if st.button("Logout / Exit", use_container_width=True):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.pop("supabase", None)
            st.session_state.logged_in = False
            st.session_state.is_guest = False
            st.session_state.user_info = None
            st.session_state.auth_mode = None
            st.rerun()

    # --- MAIN PAGE CONTENT (title and welcome show even without an API key) ---
    st.title("AWTA: Agentic Wind Tunnel Analysis")

    if st.session_state.is_guest:
        st.info("You are using the app as a guest. History will not be saved.")
    elif st.session_state.logged_in:
        st.success(f"Welcome back, {st.session_state.user_info['username']}!")

    if not user_api_key:
        st.info("Please provide a TypeSafe API Key to continue.")
        st.stop()

    # Initialize session state variables for unknown fallback
    if "show_unknown_warning" not in st.session_state:
        st.session_state.show_unknown_warning = False
    if "show_llm_info" not in st.session_state:
        st.session_state.show_llm_info = False

    files = [f for f in os.listdir("data") if f.endswith('.csv')] if os.path.exists("data") else []
    ui_files = [f for f in files if f != "xc_locations.csv"]

    with files_box:
        selected_files = st.multiselect("Choose files:", ui_files)

    with query_box:
        query = st.text_area("Query:", height=150, key="query_input")

    with buttons_box:
        col_run, col_help = st.columns(2)
        run_button = col_run.button("Run Agent", use_container_width=True, type="primary")
        if col_help.button("❓ Help", use_container_width=True):
            show_help()

    if run_button:
        st.session_state.show_unknown_warning = False
        st.session_state.show_llm_info = False

    # Data Preview
    if selected_files:
        st.subheader("Dataset Preview")
        for f in selected_files:
            file_path = os.path.join("data", f)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    with st.expander(f"Data preview: {f}", expanded=False):
                        st.dataframe(df, use_container_width=True, height=400)
                except Exception as e:
                    st.error(f"Error loading {f}: {e}")
        st.divider()

    st.subheader("Agent Output")
    image = st.empty()
    plot_container = st.empty()

    # Show the previous image
    plot_path = os.path.join(current_dir, "plot.png")
    plotly_path = os.path.join(current_dir, "plot.json")

    if os.path.exists(plot_path):
        image.image(plot_path, caption="Latest Plot", use_container_width=True)

    # Fallback Warning Box
    if st.session_state.get("show_unknown_warning", False):
        with st.container(border=True):
            st.warning("No specific tool was found for your request.")
            st.write("Do you want to generate an answer using the LLM?")
            if st.button("Generate Answer"):
                st.session_state.show_llm_info = True
            if st.session_state.get("show_llm_info", False):
                st.info("The custom LLM is currently under development")

    # Agent
    if run_button:
        st.session_state.show_unknown_warning = False
        st.session_state.show_llm_info = False

        if not selected_files or not query:
            st.error("Missing file or query!")
        else:

            final_files = selected_files.copy()
            if "xc_locations.csv" in files:
                final_files.append("xc_locations.csv")

            # Delete the previous plot if it exists
            if os.path.exists(plot_path):
                os.remove(plot_path)
                image.empty()

            if os.path.exists(plotly_path):
                os.remove(plotly_path)
                image.empty()
                plot_container.empty()

            # Delete the previous runs
            folders_to_delete = ["runs", "__pycache__"]
            for folder in folders_to_delete:
                folder_path = os.path.join(current_dir, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path, ignore_errors=True)

            # --SECRET PROMPTS--
            files_str = ', '.join(final_files)
            prefix = f"You have the following datasets loaded: {files_str}. "
            suffix = "\nCRITICAL: If using Matplotlib, save the plot locally exactly as 'plot.png'. If using Plotly, save the figure as JSON exactly as 'plot.json' using fig.write_json('plot.json'). Do NOT use plt.show() or fig.show() and do NOT print any dataframes. Execute silently. "
            final_query = prefix + query + suffix

            command = [
                sys.executable, "-u", "agent.py",
                "--data-files"
            ]

            for f in final_files:
                command.append(f)

            command.extend(["--query", final_query])

            st.toast("Agent started! Please wait...")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env={**os.environ, "TYPESAFE_API_KEY": user_api_key},
                cwd=current_dir,
                bufsize=1
            )

            stdout, _ = process.communicate()
            return_code = process.returncode

            # Find the selected tool
            selected_tool = None
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("SELECTED_TOOL:"):
                    selected_tool = line.split(":", 1)[1].strip()

            # Log query to Supabase if the user is authenticated
            if st.session_state.logged_in:
                try:
                    supabase.table("queries").insert({
                        "user_id": st.session_state.user_info["id"],
                        "query_text": query,
                        "selected_tool": selected_tool,
                    }).execute()
                except Exception as e:
                    print("Failed to log query to DB:", e)

            if return_code == 0:
                # Parse agent output by locating start and end markers
                # (the SELECTED_TOOL marker line is filtered out)
                lines = [
                    line.strip() for line in stdout.splitlines()
                    if line.strip() and not line.strip().startswith("SELECTED_TOOL:")
                ]
                start_idx = -1
                end_idx = -1
                for idx, line in enumerate(lines):
                    if line.startswith("Starting agent with query:"):
                        start_idx = idx
                    elif line == "Agent finished successfully.":
                        end_idx = idx

                agent_output = ""
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    agent_output = "\n".join(lines[start_idx + 1: end_idx]).strip()
                else:
                    agent_output = stdout.strip()

                if "unknown" in agent_output.lower():
                    st.session_state.show_unknown_warning = True
                    st.rerun()
                else:
                    st.toast("Process completed successfully!")
                    if os.path.exists(plotly_path):
                        fig = pio.read_json(plotly_path)
                        plot_container.plotly_chart(fig, use_container_width=True)
                    elif os.path.exists(plot_path):
                        image.image(plot_path, caption="New Plot Generated!", use_container_width=True)
                    else:
                        if "Error:" in agent_output:
                            error_line = next(
                                (line for line in agent_output.splitlines() if "Error:" in line),
                                "Unknown error",
                            )
                            st.error(error_line)
                        else:
                            st.error("Script finished but plot didn't generate.")
                            with st.expander("Debug: Raw Output"):
                                st.code(f"Agent Output: '{agent_output}'\n\nFull Stdout:\n{stdout}")
            else:
                st.error(f"Agent stopped with error code: {return_code}")
                with st.expander("Show Crash Logs"):
                    st.code(stdout)

    # --- HISTORY (must stay at this indentation, at the very end) ---
    with history_box:
        render_history()