import streamlit as st
from audiorecorder import audiorecorder
import datetime
import json
from google.oauth2 import service_account
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image
import pytz
import unicodedata
import re
import librosa
import numpy as np
import soundfile as sf # Import soundfile
import noisereduce as nr
import pandas as pd
import parselmouth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time

service_account_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/drive.file'])
drive_folder_id = st.secrets["DRIVE_FOLDER_ID"]  # Get from Streamlit secrets
service = build('drive', 'v3', credentials=creds)
vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')

def extract_info(input_string):
    parts = input_string.split("_")  
    name = parts[0]
    gender = parts[1]
    age = 2025 - int(parts[2])

    return {
        "name": name,
        "gender": 1 if gender == 'Nam' else 0,
        "age": age,
        "yod": 0,
        "status": 0
    }

def sort_dataframe_by_columns(df, columns=None, ascending=True):
    if columns is None:
        columns = df.columns.tolist()  # Sort by all columns

    sorted_df = df.sort_values(by=columns, ascending=ascending)
    return sorted_df

def preprocess_audio(audio_file, target_sr=48000, noise_reduction=True, silence_removal=True, target_bit_depth=16):
    try:
        y, sr = librosa.load(audio_file, sr=None)  # Load with original sampling rate

        # Resample to target sampling rate
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Noise Reduction
        if noise_reduction:
            # Using librosa's noise reduction (can be improved with more advanced methods)
            # y = librosa.effects.preemphasis(y) # add preemphasis

            # y_harmonic, y_percussive = librosa.effects.hpss(y) # separate harmonic and percussive
            # y = y_harmonic # only take harmonic to reduce noise.
            # or use more advanced noise reduction.
            # y = nr.reduce_noise(y=y, sr=sr)

            # Apply noise reduction (example using a simple method)
            # y = librosa.effects.reduce_noise(y=y, sr=target_sr)
            # Apply noise reduction (using denoise instead of reduce_noise)
            # y = librosa.effects.denoise(y=y, sr=target_sr)
            # Apply noise reduction using noisereduce
            y = nr.reduce_noise(y=y, sr=target_sr)

        # Convert to the target bit depth
        if target_bit_depth == 16:
            y = np.int16(y * 32767)

        # Silence Removal
        if silence_removal:
            y, index = librosa.effects.trim(y, top_db=20)  # Adjust top_db as needed

        return y, sr

    except Exception as e:
        print(f"Error preprocessing {audio_file}: {e}")
        return None
    
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        y, _ = librosa.effects.trim(y)

        f0 = librosa.yin(y, fmin=75, fmax=500)
        Fo = np.nanmean(f0)
        Fhi = np.nanmax(f0)
        Flo = np.nanmin(f0)

        # rms = librosa.feature.rms(y=y)[0]

        spectral_centroid = np.nanmean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.nanmean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.nanmean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_rolloff = np.nanmean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mfccs = np.nanmean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)

        # # Standardize MFCCs before PCA
        # scaler = StandardScaler()
        # mfccs_scaled = scaler.fit_transform(mfccs.reshape(1, -1)) #reshaping for scaling
        # pca = PCA(n_components=1)
        # features_reduced = pca.fit_transform(mfccs_scaled)

        f0_array = f0
        f0_mean = np.nanmean(f0_array)
        f0_std = np.nanstd(f0_array)
        f0_skewness = np.nanmean((f0_array - f0_mean)**3) / (f0_std**3)
        f0_kurtosis = np.nanmean((f0_array - f0_mean)**4) / (f0_std**4)

        file_name = os.path.basename(audio_file)
        info = extract_info(file_name)

        features = {
            "file": file_name,
            "name": info["name"],
            "gender": info["gender"],
            "age": info["age"],
            "yod": info["yod"],
            "status": info["status"],
            "MDVP:Fo(Hz)": Fo,
            "MDVP:Fhi(Hz)": Fhi,
            "MDVP:Flo(Hz)": Flo,
            # "RMS": rms,
            "Spectral_Centroid": spectral_centroid,
            "Spectral_Bandwidth": spectral_bandwidth,
            "Spectral_Contrast": spectral_contrast,
            "Spectral_Rolloff": spectral_rolloff,
            "MFCC_1": mfccs[0],
            "MFCC_2": mfccs[1],
            "MFCC_3": mfccs[2],
            "MFCC_4": mfccs[3],
            "MFCC_5": mfccs[4],
            "MFCC_6": mfccs[5],
            "MFCC_7": mfccs[6],
            "MFCC_8": mfccs[7],
            "MFCC_9": mfccs[8],
            "MFCC_10": mfccs[9],
            "MFCC_11": mfccs[10],
            "MFCC_12": mfccs[11],
            "MFCC_13": mfccs[12],
            "MFCC_14": mfccs[13],
            "MFCC_15": mfccs[14],
            "MFCC_16": mfccs[15],
            "MFCC_17": mfccs[16],
            "MFCC_18": mfccs[17],
            "MFCC_19": mfccs[18],
            "MFCC_20": mfccs[19],
            # "PCA_1": features_reduced[0][0],
            "F0_Mean": f0_mean,
            "F0_Std": f0_std,
            "F0_Skewness": f0_skewness,
            "F0_Kurtosis": f0_kurtosis,
        }
        return features
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None
    
def predict_pd(audio, _name, _gender, _year_of_birth, _phone):
    st.audio(audio.export().read())
    
    utc_now = datetime.datetime.now().replace(tzinfo=pytz.utc)
    vietnam_now = utc_now.astimezone(vietnam_timezone)
    timestamp = vietnam_now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    __gender = _gender
    if unicodedata.normalize("NFC", _gender) == "Nữ":
        __gender = "Nu"
    filename = f"{_name}_{__gender}_{_year_of_birth}_{_phone}_{timestamp}_a.wav"

    audio.export(filename, format="wav")
    print(filename)
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

    preprocessed_audio = preprocess_audio(filename)

    output_file_path = ""
    if preprocessed_audio is not None:
        y, sr = preprocessed_audio
        output_file_path = filename.replace(".wav", "_harmonized.wav")
        # Use soundfile.write instead of librosa.output.write_wav
        sf.write(output_file_path, y, sr)
        print(f"Preprocessed {filename} and saved to {output_file_path}")
    else:
        print(f"Skipping {filename} due to errors.")

    all_features = []
    features = extract_features(output_file_path)
    if features:
        all_features.append(features)
        print(f"Extracted features for {filename}")
        print(features)
    else:
        print(f"Skipping {filename} due to errors.")
    df = pd.DataFrame(all_features)
    print(df)
    # df.drop(['file','name'], axis=1, inplace=True)
    df = df.iloc[:, 2:]  # Keeps only columns from index 2 onwards
    print(df)
    # biomaker to keep
    candidates = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    df = df.iloc[:, candidates]
    print('after keep biomaker:')
    print(df)
    # Save the model and scaler
    model_filename = 'logistic_regression_model.joblib'
    scaler_filename = 'scaler.joblib'

    # Load the model and scaler later
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)

    # Predict
    npy_arr = df.to_numpy()
    print('npy_arr:')
    print(npy_arr)
    new_data_scaled = loaded_scaler.transform(npy_arr)
    predictions = loaded_model.predict(new_data_scaled)
    print("\nPredictions using loaded model:\n", predictions)

    file_metadata = {
        'name': filename,
        'parents': [drive_folder_id]
    }

    media = MediaFileUpload(filename, mimetype='audio/wav')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Ghi âm '{filename}' đã được lưu vào Google Drive")
    print(f"File ID: {file.get('id')}")

    # file_metadata = {
    #     'name': output_file_path,
    #     'parents': [drive_folder_id]
    # }

    # media = MediaFileUpload(output_file_path, mimetype='audio/wav')
    # file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    # print(f"Ghi âm '{output_file_path}' đã được lưu vào Google Drive")
    # print(f"File ID: {file.get('id')}")

    # Clean up the local file after upload
    os.remove(filename)
    os.remove(output_file_path)
    return predictions

st.markdown(
    """
    <link rel="icon" href="favicon.ico" type="image/x-icon">
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .st-emotion-cache-1v0mbdj.e115fcil1 {display: none;}  /* Streamlit Cloud profile footer */
        h1 {
            font-size: 24px;  /* Adjust the size here */
        }
        [data-testid="stColumn"] {
            padding: 0px !important;
        }
        label {
            font-size: 14px; /* Reduced label font size */
            margin-bottom: 2px; /* Reduced margin */
        }
        input, [data-baseweb="input"], [data-baseweb="input-container"] {
            font-size: 14px; /* Reduced input font size */
            padding: 4px; /* Reduced padding */
            margin-bottom: 4px; /* Reduced margin */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

logo = Image.open("logo_new.png")
col1a, col2a = st.columns([2, 4])  # Điều chỉnh tỷ lệ cột tùy ý
with col1a:
    st.image(logo, width=300)
with col2a:
    st.subheader("CHẨN ĐOÁN BỆNH PARKINSON QUA GIỌNG NÓI")
st.write("""
         Giới thiệu: đây là 1 đồ án nghiên cứu, giọng nói ông/bà cô/chú anh/chị sẽ được lưu lại nhằm mục đích nâng cao kết quả nghiên cứu.
         """)
st.markdown("THÔNG TIN CÁ NHÂN:")

col1, col2 = st.columns([1, 2])
with col1:
    st.write("Họ tên:")
with col2:
    name = st.text_input("name_input", key="name_input", label_visibility="collapsed")

col7, col8 = st.columns([1, 2])
with col7:
    st.write("Giới tính:")
with col8:
    gender = st.radio("gender_input", ['Nam', 'Nữ'], key="gender_input", label_visibility="collapsed")

col3, col4 = st.columns([1, 2])
with col3:
    st.write("Năm sinh:")
with col4:
    year_of_birth = st.number_input("yob_input", value=1960, min_value=1900, max_value=2025, step=1, key="yob_input", label_visibility="collapsed")

# col5, col6 = st.columns([1, 2])
# with col5:
#     st.write("Số điện thoại:")
# with col6:
#     phone = st.text_input("phone_input", key="phone_input", label_visibility="collapsed")
#     # Kiểm tra nếu người dùng đã nhập gì đó
#     if phone:
#         # Regex kiểm tra số điện thoại VN bắt đầu bằng 0 và có 10 chữ số
#         if re.fullmatch(r"0\d{9}", phone):
#             st.success("Số điện thoại hợp lệ!")
#         else:
#             st.error("Số điện thoại không hợp lệ. Vui lòng nhập đúng định dạng.")
# st.write("""
#          Ghi chú: Số điện thoại sẽ được dùng để liên hệ lại sau 1 khoảng thời gian 6 tháng hoặc 1 năm để xác nhận tình trạng bệnh nhằm bổ sung thông tin vào dữ liệu nghiên cứu.
#          """)
phone = '0908123456'

# Khởi tạo trạng thái
if "recording" not in st.session_state:
    st.session_state.recording = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

st.markdown("---")
st.markdown("NỘI DUNG CHẨN ĐOÁN:")

# st.write("Mẫu ghi âm như sau (phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá...):")
# # Mở file âm thanh
# audio_file = open('Aaaa_sample.wav', 'rb')
# # Hiển thị audio player
# st.audio(audio_file, format='audio/wav')

st.write("Hít nhẹ và phát âm nguyên âm “A” thật to, đều, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá...")
audio1 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam1")

# # Khi bắt đầu ghi
# if audio1 is None and not st.session_state.recording:
#     st.session_state.recording = True
#     st.session_state.start_time = time.time()

# # Khi ngừng ghi
# if audio1 is not None and st.session_state.recording:
#     st.session_state.recording = False
#     duration = int(time.time() - st.session_state.start_time)
#     st.success(f"Đã ghi âm xong! Thời lượng: {duration} giây")

# # Hiển thị trạng thái đang ghi
# if st.session_state.recording:
#     elapsed = int(time.time() - st.session_state.start_time)
#     st.info(f"🎤 Đang ghi âm... {elapsed} giây")

if len(audio1) > 0:
    with st.spinner("Đang phân tích..."):
        predict = predict_pd(audio1, name, gender, year_of_birth, phone)
        print(f"Predict: {predict}")
        if predict[0] == 0:
            st.success("Kết quả chẩn đoán: Xác suất bị bệnh thấp")
        else:
            st.success("Kết quả chẩn đoán: Xác suất bị bệnh cao")
# st.write("2. Nghỉ 1 chút, hít nhẹ và phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá... (lần 2)")
# audio2 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam2")
# if len(audio2) > 0:
#     save_ggdrive(audio2, name, gender, year_of_birth, phone)
# st.write("3. Nhỉ 1 chút nữa, hít nhẹ và phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá... (lần 3)")
# audio3 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam3")
# if len(audio3) > 0:
#     save_ggdrive(audio3, name, gender, year_of_birth, phone)
st.markdown("---")
st.write("Lời cảm ơn: Xin cảm ơn ông/bà cô/chú anh/chị Cộng Đồng PARKINTON VIỆT NAM, đặc biệt là anh admin Tung Mix vì đã hỗ trợ em thực hiện đồ án này!")
logo2 = Image.open("logo2.png")
st.image(logo2)
