# Dashboard Launch Instructions

## Option 1: Direct Launch (Recommended)

```bash
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
streamlit run eval_dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

Or use the convenience script:
```bash
cd /workspace-vast/annas/git/research-tools
./eval_dashboard/run_dashboard.sh
```

The dashboard will be available at: `http://<your-server-ip>:8501`

## Option 2: SSH Port Forwarding

If you're accessing the server via SSH, forward the port:

```bash
# On your local machine
ssh -L 8501:localhost:8501 user@server-address
```

Then launch the dashboard on the server:
```bash
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
streamlit run eval_dashboard/app.py
```

Access at: `http://localhost:8501` on your local machine

## Option 3: VSCode Port Forwarding

If using VSCode Remote:
1. Launch the dashboard (as in Option 1)
2. VSCode will automatically detect port 8501
3. Click "Open in Browser" when prompted
4. Or manually forward the port in the "Ports" tab

## Usage

Once the dashboard loads:

1. **Sidebar Controls**:
   - Verify the data path points to the preprocessed file
   - Select a probe type (orthogonal_raw, text_raw, centroid_k10)
   - Choose a conversation from the dropdown
   - Select emotions to visualize (default: anger, fear, sadness)

2. **Individual Tab**: View emotion trajectories for a single conversation
   - Interactive plot with sentence-level resolution
   - Background shading shows user (blue) vs assistant (yellow) turns
   - Hover for details
   - Expand to see annotated conversation text

3. **Aggregated Tab**: Statistical analysis across all conversations
   - Mean trajectories with 95% confidence intervals
   - Statistical summary by conversation phase
   - Heatmap visualization
   - CSV export

4. **Probe Comparison Tab**: Coming soon!

## Troubleshooting

### "Data file not found"
Run the preprocessing first:
```bash
sbatch eval_dashboard/slurm_preprocess_dashboard.sh
# Or run locally (requires GPU):
python eval_dashboard/data_preprocessing.py --input <input_file> --output <output_file>
```

### Port already in use
Change the port:
```bash
streamlit run eval_dashboard/app.py --server.port 8502
```

### Module not found errors
Install dependencies:
```bash
cd /workspace-vast/annas/git/research-tools
uv pip install streamlit plotly pandas sentences
```
