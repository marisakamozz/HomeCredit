import pathlib
import numpy as np
import pandas as pd 
import pandas_profiling

for file in pathlib.Path('../input').glob('*.csv'):
    if file.name != 'HomeCredit_columns_description.csv':
        df = pd.read_csv(file)
        profile = df.profile_report(title=f'Pandas Profiling Report - {file.name}')
        profile.to_file(output_file=f'../logs/report/{file.stem}.html')
