import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "/Users/chocosongee/Desktop/cell/large_b8e100.pt"

def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color=(255,0,0), alpha=0.4):
    image_np = np.array(image).copy()
    red, green, blue = color

    # 마스크가 True 또는 1인 픽셀에 빨간색 반투명 덮기
    image_np[mask > 0, 0] = (1 - alpha) * image_np[mask > 0, 0] + alpha * red
    image_np[mask > 0, 1] = (1 - alpha) * image_np[mask > 0, 1] + alpha * green
    image_np[mask > 0, 2] = (1 - alpha) * image_np[mask > 0, 2] + alpha * blue

    return Image.fromarray(image_np.astype(np.uint8))

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

def main():
    st.title("YOLO Segmentation 빨간 마스크 시각화")

    model = load_model()

    uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="원본 이미지", use_column_width=True)

        results = model(image)

        result = results[0]  # 첫 번째 결과 가져오기

        if result.masks is not None:
            mask_tensor = result.masks.data[0].cpu().numpy()
            mask_bool = mask_tensor > 0.5

            resized_mask = np.array(Image.fromarray(mask_bool.astype(np.uint8)*255).resize(image.size)) > 128

            overlayed_img = overlay_mask_on_image(image, resized_mask, color=(255,0,0), alpha=0.4)
            st.image(overlayed_img, caption="Segmentation 마스크 빨간색 오버레이", use_column_width=True)
        else:
            st.write("Segmentation 마스크가 없습니다.")

if __name__ == "__main__":
    main()
