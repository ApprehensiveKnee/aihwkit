import os
import pandas as pd
import glob

def get_drift_time(directory):
    """Extract drift time from directory name."""
    d = directory.split('d=')[1]
    if ',' not in d:
        return f"{d}"
    time = d.split('d,')[0]
    temp = d.split('d,')[1]
    return f"{time} day, {temp}"

def format_size(size):
    """Format matrix size for LaTeX."""
    per_char ="\\times"
    return f"${size.replace('x', per_char)}$"

def create_latex_table():
    # Table header
    latex = r"""\begin{table}[htb]
    \centering    
    \caption{Comparisons between MAC approximations\label{tab:tile_performances}}
    \begin{tabular}{l l l  c c c c}
    \toprule
    Drift & Level & Size & \multicolumn{4}{c}{Compensation type} \\
    \cmidrule(lr){4-7}
    & & & NO COMP. & IBM & DIFF & DIFF+IBM \\
    \midrule
"""
    
    # Get all directories starting with 'd='
    drift_dirs = ["d=Prog","d=1d,RT", "d=30d,RT", "d=1d,60°C", "d=1d,90°C", "d=1d,120°C", "d=1d,150°C", "d=1d,180°C"]
    
    for drift_dir in drift_dirs:
        drift_time = get_drift_time(drift_dir)
        
        # Get all level directories
        level_dirs = sorted(glob.glob(os.path.join(drift_dir, 'lv=*')))
        
        first_level = True
        for level_dir in level_dirs:
            level = level_dir.split('lv=')[1]

            if not first_level:
                latex += "    \\cmidrule(lr){2-7}\n"
            
            # Read CSV file
            csv_files = glob.glob(os.path.join(level_dir, '*.csv'))
            if not csv_files:
                continue
            df = pd.read_csv(csv_files[0])
            
            # Matrix sizes
            sizes = ['128x64', '256x128', '256x256', '512x256', '512x512']
            
            for i, size in enumerate(sizes):
                if i == 1:
                    continue
                row_data = df.iloc[i]  # Skip header row
                
                # First row of each drift section
                if first_level and i == 0:
                    latex += f"    {drift_time} & {level} & {format_size(size)}"
                # First row of other levels
                elif i == 0:
                    latex += f"     & {level} & {format_size(size)}"
                # Subsequent rows
                else:
                    latex += f"     &  & {format_size(size)}"
                
                # Add data columns, using scientific notation
                # also use bold for the best value
                for comp_type in ['NO_COMP', 'IBM', 'DIFF', 'DIFF_IBM']:
                    value = row_data[comp_type]
                    if value == row_data.min():
                        latex += f" & \\textbf{{{value:.2e}}}"
                    else:
                        latex += f" & {value:.2e}"
                latex += r" \\"
                latex += "\n"
            
            first_level = False
        
        if drift_dir != drift_dirs[-1]:
            latex += "    \\midrule\n"

        
    
    # Table footer
    latex += r"""    \bottomrule
    \end{tabular}
\end{table}
"""
    
    return latex

# move to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Generate and save the LaTeX table
with open('performance_table.tex', 'w') as f:
    f.write(create_latex_table())
    print(f"Table saved as performance_table.tex in {os.getcwd()}")