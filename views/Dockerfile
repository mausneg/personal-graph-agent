FROM python:3.10-slim

WORKDIR /app

RUN pip install streamlit python-dotenv

COPY views/ views/
EXPOSE 8502

CMD ["streamlit", "run", "views/chat.py", "--server.port=8502", "--server.address=0.0.0.0"]