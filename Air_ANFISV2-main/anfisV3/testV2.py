

def run_test(filepath, epoch_n, mf, step_size, decrease_rate, increase_rate, log_widget,tk,np,dd,MinMaxScaler,anfis,log_message):

    # อ่านไฟล์ CSV ด้วย Dask (จะไม่โหลดข้อมูลทั้งหมดในหน่วยความจำในครั้งเดียว)
    # df = dd.read_csv(filepath)


    df = dd.read_csv(filepath, header=None, names=['Time', 'Flow', 'Pressure'])

    # Compute min and max for each column
    min_values = df.min().compute()
    max_values = df.max().compute()

    # Separate into individual variables
    time_min = min_values['Time']
    flow_min = min_values['Flow']
    pressure_min = min_values['Pressure']

    time_max = max_values['Time']
    flow_max = max_values['Flow']
    pressure_max = max_values['Pressure']

    # # Print for confirmation
    # print(f"Time Min: {time_min}, Max: {time_max}")
    # print(f"Flow Min: {flow_min}, Max: {flow_max}")
    # print(f"Pressure Min: {pressure_min}, Max: {pressure_max}")

    # คำนวณข้อมูลที่โหลดจาก Dask (compute จะทำให้ข้อมูลเป็น Pandas DataFrame)
    data = df.compute().values

    #data = np.genfromtxt(filepath, delimiter=',')
    
    # Divide data into input and output
    inputs = data[:, :-1]  # All columns except the last one are inputs
    output = data[:, -1:]  # The last column is the output
    ndata = data.shape[0]  # Data length


    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_input = scaler.fit_transform(inputs)

    # input1_min, input1_max = np.min(inputs[:, 0]), np.max(inputs[:, 0])
    # input2_min, input2_max = np.min(inputs[:, 1]), np.max(inputs[:, 1])

    # ANFIS train
    bestnet, y_myanfis, RMSE = anfis.myanfis(data, inputs, epoch_n, mf, step_size, decrease_rate, increase_rate,log_message )

    y_myanfis = anfis.evalmyanfis(bestnet, inputs)

    anfis_predictions = y_myanfis

    # For classification problem ( Round outputs to int)
    anfis_predictions = np.round(anfis_predictions).astype(int)

    # Calculate the RMSE
    rmse = anfis.calc_rmse(output,anfis_predictions)

    msg = f'Total RMSE error myanfis: {rmse:.2f}'
    print(msg)  # Print the message

    log_message(msg)

    anfis.print_membership_functions(bestnet,log_message)

    

    return bestnet, data, output, anfis_predictions


