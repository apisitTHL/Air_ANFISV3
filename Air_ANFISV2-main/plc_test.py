import numpy as np

def gbellmf(x, a, b, c):
    if a == 0:
        return 0
    else:
        tmp = (x - c) / a
        return 1 / (1 + np.exp(2 * b * np.log(np.abs(tmp))))

# Define inputs
hour = 0
minute = 5
flow = 2705.6499


input1 = (hour*60 + minute)*0.000694774  # Example value
input2 = flow  # Example value

# Define Membership Function parameters
MF_Params = np.array([
    [[0.18951922054595577, 2.0150305045426804, 0.15748846328563104],
     [-0.1281057756976041, 2.0230029523151343, 0.6147683969306972],
     [0.027700099076349405, 2.0149565885652607, 1.0582546020165513]],  # Input1 MF1-MF3
    [[2256.4828745110194, 2.2299610024787935, -0.000282374594764847],
     [2256.483648757557, 1.90008687192553, 6769.449693605185],
     [2256.483402011631, 1.9515623913090394, 13538.899964388316]]   # Input2 MF1-MF3
])

# Define Consequent parameters (kparams)
Kparams = np.array([[ 3.96079296e-03,  2.00411960e-04, 1.41829906e+00],
 [-4.25208508e-04,  2.00171802e-04 , 1.41780868e+00],
 [ 3.89613234e-04 , 2.00145510e-04 , 1.41713497e+00],
 [-2.74805276e-05  ,2.00011363e-04 , 1.41900903e+00],
 [-4.50256567e-04 , 2.00023783e-04 , 1.41888116e+00],
 [ 3.30639092e-05 , 1.99917022e-04 , 1.41990977e+00],
 [-2.20655865e-02 , 1.94818321e-04 , 1.43989163e+00],
 [-1.09122293e-01  ,1.94548983e-04 , 1.57890199e+00],
 [-1.79240078e-01 , 1.99692523e-04,  1.60114141e+00]])

# Step 1: Calculate Membership values
MF_VAL = np.zeros((2, 3))
for i in range(3):
    MF_VAL[0, i] = gbellmf(input1, *MF_Params[0, i])
    MF_VAL[1, i] = gbellmf(input2, *MF_Params[1, i])

# Step 2: Calculate Firing Strength
FS = np.zeros(9)
FS[0] = MF_VAL[0,0] * MF_VAL[1,0]
FS[1] = MF_VAL[0,0] * MF_VAL[1,1]
FS[2] = MF_VAL[0,0] * MF_VAL[1,2]
FS[3] = MF_VAL[0,1] * MF_VAL[1,0]
FS[4] = MF_VAL[0,1] * MF_VAL[1,1]
FS[5] = MF_VAL[0,1] * MF_VAL[1,2]
FS[6] = MF_VAL[0,2] * MF_VAL[1,0]
FS[7] = MF_VAL[0,2] * MF_VAL[1,1]
FS[8] = MF_VAL[0,2] * MF_VAL[1,2]

# Step 3: Normalize Firing Strength
FS_SUM = np.sum(FS)
NORM_FS = FS / FS_SUM if FS_SUM != 0 else np.zeros(9)

# Step 4: Sugeno Output Calculation
Rule_Out = NORM_FS * (Kparams[:,0] * input1 + Kparams[:,1] * input2 + Kparams[:,2])

# Step 5: Final Output
ANFIS_OUTPUT = np.sum(Rule_Out)

# Print Results
# print("Membership Values:", MF_VAL)
# print("Firing Strengths:", FS)
# print("Normalized Firing Strengths:", NORM_FS)
# print("Rule Outputs:", Rule_Out)
print("Time:", hour,":",minute)
print("Time_convert:", input1)
print("Final ANFIS Output:", ANFIS_OUTPUT)
