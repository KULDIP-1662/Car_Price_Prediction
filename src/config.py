
#raw data path
RAW_DATAPATH = "data/raw/vehicles.csv"

#cleaned data path
CLEANED_DATAPATH = "data/processed/cleaned_car_csv.csv"

#train test val paths
XTRAIN_PATH = "data/splited/x_train.csv"
XTEST_PATH = "data/splited/x_test.csv"
YTRAIN_PATH = "data/splited/y_train.csv"
YTEST_PATH = "data/splited/y_test.csv"
XVAL_PATH = "data/splited/x_val.csv"
YVAL_PATH = "data/splited/y_val.csv"

#model path
MODEL_PATH = r"C:\Users\Kulde\OneDrive\Desktop\OFFICE\Car_Price_Prediction\models\model.pth"

#scaler and encoder paths
SCALER_X_PATH = r"C:\Users\Kulde\OneDrive\Desktop\OFFICE\Car_Price_Prediction\models\scaler_x.pkl"
SCALER_Y_PATH = r"C:\Users\Kulde\OneDrive\Desktop\OFFICE\Car_Price_Prediction\models\scaler_y.pkl"
ENCODER_PATH = r"C:\Users\Kulde\OneDrive\Desktop\OFFICE\Car_Price_Prediction\models\encoder.pkl"
#parameteres for neural network
EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01