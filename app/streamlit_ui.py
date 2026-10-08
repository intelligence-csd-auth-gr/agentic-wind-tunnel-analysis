import streamlit as st
import subprocess
import os
import shutil
import sys
import pandas as pd
import plotly.io as pio


current_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="AWTA", layout="wide")
st.title("AWTA: Agentic Wind Tunnel Analysis")

st.sidebar.header("Authentication")
user_api_key = st.sidebar.text_input("Gemini API Key:", type="password")

if not user_api_key:
    st.info("Please provide a Gemini API Key to continue.")
    st.stop()

os.environ["GEMINI_API_KEY"]=user_api_key

# Initialize session state variables for unknown fallback
if "show_unknown_warning" not in st.session_state:
    st.session_state.show_unknown_warning = False
if "show_llm_info" not in st.session_state:
    st.session_state.show_llm_info = False



files = [f for f in os.listdir("data") if f.endswith('.csv')] if os.path.exists("data") else []
ui_files=[f for f in files if f != "xc_locations.csv"]

#Left Bar
st.sidebar.header("Settings")
selected_files = st.sidebar.multiselect("Choose files:", ui_files)
query = st.sidebar.text_area("Query:", height=150)
run_button = st.sidebar.button("Run Agent")

if run_button:
    st.session_state.show_unknown_warning = False
    st.session_state.show_llm_info = False

#Data Preview
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
image=st.empty() 
plot_container = st.empty()

#Show the previus image
plot_path=os.path.join(current_dir, "plot.png")
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



#Agent
if run_button:
    st.session_state.show_unknown_warning = False
    st.session_state.show_llm_info = False
    
    if not selected_files or not query:
        st.error("Missing file or query!")
    else:

        final_files = selected_files.copy()
        if "xc_locations.csv" in files:
            final_files.append("xc_locations.csv")

        #Delete the previus plot if exists
        if os.path.exists(plot_path):
            os.remove(plot_path)
            image.empty()
        
        if os.path.exists(plotly_path):
            os.remove(plotly_path)
            image.empty()
            plot_container.empty()
        
        
        #Delete the previus runs
        folders_to_delete = ["runs", "__pycache__"]
        for folder in folders_to_delete:
            folder_path = os.path.join(current_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
        
        #--SECRET PROMPTS--
        files_str=', '.join(final_files)
        prefix=f"You have the following datasets loaded: {files_str}. "
        suffix="\nCRITICAL: If using Matplotlib, save the plot locally exactly as 'plot.png'. If using Plotly, save the figure as JSON exactly as 'plot.json' using fig.write_json('plot.json'). Do NOT use plt.show() or fig.show() and do NOT print any dataframes. Execute silently. "
        final_query=prefix+query+suffix

        #terminal_output = st.empty()
        full_log = ""
        
        
        command = [
            sys.executable, "-u", "agent.py", 
            "--data-files"
        ]

        for f in final_files:
            command.append(f)

        command.extend(["--query", final_query])
        
        #print(command)
        
        st.toast("Agent started! Please wait...")
       

        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8',
            env=os.environ,
            cwd=current_dir, 
            bufsize=1       
        )

       
        
        stdout, _=process.communicate()
        return_code=process.returncode

        if return_code==0:
            # Parse agent output by locating start and end markers
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            start_idx = -1
            end_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith("Starting agent with query:"):
                    start_idx = idx
                elif line == "Agent finished successfully.":
                    end_idx = idx

            agent_output = ""
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                agent_output = "\n".join(lines[start_idx + 1 : end_idx]).strip()
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
                    image.image(plot_path, caption="New Plot Generated!",use_container_width=True)
                else:
                    if "Error:" in agent_output:
                        error_line = next((line for line in agent_output.splitlines() if "Error:" in line))
                        st.error(error_line)
                    else:
                        st.error("Script finished but plot didnt generated.")
                        with st.expander("Debug: Raw Output"):
                            st.code(f"Agent Output: '{agent_output}'\n\nFull Stdout:\n{stdout}")
        else:
            st.error(f"Agent stopped with error code: {return_code}")
            with st.expander("Show Crash Logs"):
                st.code(stdout)