import os
import streamlit as st

import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image
from langchain_community.llms import Ollama


def convert_to_base64(pil_image, file_format="JPEG"):
    """
    Convert PIL images to Base64 encoded strings
    """

    buffered = BytesIO()
    pil_image.save(buffered, format=file_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


llava = Ollama(model="llava")

if st.sidebar.button("Home"):
    st.switch_page("app.py")
st.title("Ответы на вопросы по изображению")
st.sidebar.title("Обработка изображений")

upload_files = st.sidebar.file_uploader("Загрузить изображения", accept_multiple_files=False)


def generate_answer(image, question):
    llm_with_image_context = llava.bind(images=[image])
    answer = llm_with_image_context.stream(question)
    return answer


def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


clear_cache()

if upload_files:
    file_extention = os.path.splitext(upload_files.name)[1][1:].upper()
    print(f"расширение файла {file_extention} and {upload_files}")
    pil_image = Image.open(upload_files)
    converted_image = convert_to_base64(pil_image, file_extention)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Генерируем ответ..."):
                stream = st.write_stream(generate_answer(converted_image, prompt))
            st.markdown(stream)

        st.session_state.messages.append({"role": "assistant", "content": stream})
