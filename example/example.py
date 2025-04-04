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

service_account_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/drive.file'])
drive_folder_id = st.secrets["DRIVE_FOLDER_ID"]  # Get from Streamlit secrets
service = build('drive', 'v3', credentials=creds)
vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')

def save_ggdrive(audio, _name, _gender, _year_of_birth, _years_parkinson):
    st.audio(audio.export().read())
    
    utc_now = datetime.datetime.now().replace(tzinfo=pytz.utc)
    vietnam_now = utc_now.astimezone(vietnam_timezone)
    timestamp = vietnam_now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    __gender = _gender
    if unicodedata.normalize("NFC", _gender) == "Nữ":
        __gender = "Nu"
    filename = f"{_name}_{__gender}_{_year_of_birth}_{_years_parkinson}_{timestamp}_a.wav"

    audio.export(filename, format="wav")
    print(filename)
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

    file_metadata = {
        'name': filename,
        'parents': [drive_folder_id]
    }

    media = MediaFileUpload(filename, mimetype='audio/wav')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    st.success(f"Ghi âm '{filename}' đã được lưu vào Google Drive")
    print(f"File ID: {file.get('id')}")
    # Clean up the local file after upload
    os.remove(filename)

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

logo = Image.open("logo.png")
col1a, col2a = st.columns([1, 4])  # Điều chỉnh tỷ lệ cột tùy ý
with col1a:
    st.image(logo, width=100)
with col2a:
    st.subheader("NỘI DUNG GHI ÂM GIỌNG NÓI ĐỐI VỚI NGƯỜI BỆNH PARKINSON")
st.write("""
         Mục đích của việc ghi âm này là để thực hiện 1 đồ án nghiên cứu: giọng nói của những người bị bệnh Parkinson 
         sẽ được đối chiếu với giọng nói của những người không bị bệnh Parkinson, từ đó giúp phát hiện ra bệnh Parkinson từ giai đoạn sớm.
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

col5, col6 = st.columns([1, 2])
with col5:
    st.write("Số năm bị bệnh Parkinson:")
with col6:
    years_parkinson = st.number_input("yod_input", min_value=1, step=1, key="yod_input", label_visibility="collapsed")
st.markdown("---")
st.markdown("NỘI DUNG GHI ÂM:")
st.write("Mẫu ghi âm như sau (phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá...):")
# Mở file âm thanh
audio_file = open('Aaaa_sample.wav', 'rb')
# Hiển thị audio player
st.audio(audio_file, format='audio/wav')
st.write("1. Hít nhẹ và phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá... (lần 1)")
audio1 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam1")
if len(audio1) > 0:
    save_ggdrive(audio1, name, gender, year_of_birth, years_parkinson)
st.write("2. Nghỉ 1 chút, hít nhẹ và phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá... (lần 2)")
audio2 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam2")
if len(audio2) > 0:
    save_ggdrive(audio2, name, gender, year_of_birth, years_parkinson)
st.write("3. Nhỉ 1 chút nữa, hít nhẹ và phát âm nguyên âm “A” thật to, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá... (lần 3)")
audio3 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam3")
if len(audio3) > 0:
    save_ggdrive(audio3, name, gender, year_of_birth, years_parkinson)
st.markdown("---")
st.write("Lời cảm ơn: Xin cảm ơn ông/bà cô/chú anh/chị Cộng Đồng PARKINTON VIỆT NAM, đặc biệt là anh admin Tung Mix vì đã hỗ trợ em thực hiện đồ án này!")
