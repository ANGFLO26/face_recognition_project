from model import FaceRecognitionModel
from pymongo.errors import ConnectionFailure

# Kiểm tra kết nối MongoDB
def check_mongo_connection(face_recognition_model):
    try:
        face_recognition_model.client.admin.command("ping")
        print("✅ Kết nối MongoDB thành công!")
    except ConnectionFailure as e:
        print(f"❌ Kết nối MongoDB thất bại. Lỗi: {e}")
        return False
    return True

# Khởi tạo mô hình
face_recognition_model = FaceRecognitionModel()

# Kiểm tra kết nối trước khi tiếp tục
if not check_mongo_connection(face_recognition_model):
    exit()  # Thoát chương trình nếu không kết nối được

# Đăng ký người dùng
username = input("Nhập tên người dùng: ")
image_path = "/home/phanvantai/THE_END/datatest/132.jpg"

registration_success = face_recognition_model.register_user(username, image_path)

if registration_success:
    print(f"✅ Đăng ký thành công người dùng: {username}")

    # Xác thực người dùng
    authenticated_user = face_recognition_model.authenticate_user(image_path)
    if authenticated_user:
        print(f"✅ Người dùng xác thực: {authenticated_user}")
    else:
        print("❌ Không thể xác thực. Có thể mô hình chưa được huấn luyện đủ dữ liệu.")
else:
    print("❌ Đăng ký thất bại.")

# Kiểm tra vector người dùng
print("🔍 Danh sách vector người dùng:")
face_recognition_model.check_user_vectors()
