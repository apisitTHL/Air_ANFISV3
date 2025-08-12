import dask.dataframe as dd
import pandas as pd
import numpy as np

# ---------- Your ANFIS Parameters ----------
def gbellmf(x, a, b, c):
    if a == 0:
        return 0
    tmp = (x - c) / a
    return 1 / (1 + np.exp(2 * b * np.log(np.abs(tmp))))

MF_Params = np.array([
    [[0.18951922054595577, 2.0150305045426804, 0.15748846328563104],
     [-0.1281057756976041, 2.0230029523151343, 0.6147683969306972],
     [0.027700099076349405, 2.0149565885652607, 1.0582546020165513]],
    
    [[2256.4828745110194, 2.2299610024787935, -0.000282374594764847],
     [2256.483648757557, 1.90008687192553, 6769.449693605185],
     [2256.483402011631, 1.9515623913090394, 13538.899964388316]]
])

Kparams = np.array([[ 3.96079296e-03,  2.00411960e-04, 1.41829906e+00],
 [-4.25208508e-04,  2.00171802e-04 , 1.41780868e+00],
 [ 3.89613234e-04 , 2.00145510e-04 , 1.41713497e+00],
 [-2.74805276e-05  ,2.00011363e-04 , 1.41900903e+00],
 [-4.50256567e-04 , 2.00023783e-04 , 1.41888116e+00],
 [ 3.30639092e-05 , 1.99917022e-04 , 1.41990977e+00],
 [-2.20655865e-02 , 1.94818321e-04 , 1.43989163e+00],
 [-1.09122293e-01  ,1.94548983e-04 , 1.57890199e+00],
 [-1.79240078e-01 , 1.99692523e-04,  1.60114141e+00]])

# ---------- ANFIS Predictor ----------
def anfis_predict(time_value, flow_value):
    input1 = time_value
    input2 = flow_value

    print(MF_Params)

    MF_VAL = np.zeros((2, 3))
    for i in range(3):
        MF_VAL[0, i] = gbellmf(input1, *MF_Params[0, i])
        MF_VAL[1, i] = gbellmf(input2, *MF_Params[1, i])

    FS = np.array([
        MF_VAL[0,0]*MF_VAL[1,0], MF_VAL[0,0]*MF_VAL[1,1], MF_VAL[0,0]*MF_VAL[1,2],
        MF_VAL[0,1]*MF_VAL[1,0], MF_VAL[0,1]*MF_VAL[1,1], MF_VAL[0,1]*MF_VAL[1,2],
        MF_VAL[0,2]*MF_VAL[1,0], MF_VAL[0,2]*MF_VAL[1,1], MF_VAL[0,2]*MF_VAL[1,2]
    ])

    FS_SUM = np.sum(FS)
    NORM_FS = FS / FS_SUM if FS_SUM != 0 else np.zeros(9)

    Rule_Out = NORM_FS * (Kparams[:,0]*input1 + Kparams[:,1]*input2 + Kparams[:,2])
    return np.sum(Rule_Out)

# ---------- Read your CSV ----------
# Assume CSV has: hour, minute, flow (no header)
df = dd.read_csv('Book8 (1).csv', header=None, names=['Time', 'Flow', 'Pressure'])

# Compute DataFrame and convert Time to actual input
data = df.compute()
data['Time_Converted'] = data['Time'] * 1  # already in minutes * constant

# Run ANFIS for each row
data['ANFIS_Output'] = data.apply(lambda row: anfis_predict(row['Time'], row['Flow']), axis=1)

# Save to CSV
data.to_csv("ANFIS_results.csv", index=False)

print("✅ Done! Output saved to ANFIS_results.csv")
