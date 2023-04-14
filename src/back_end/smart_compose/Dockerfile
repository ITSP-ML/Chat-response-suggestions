FROM python:3.9
# 
WORKDIR /code
# export port for nginx
EXPOSE 8002


# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./src/back_end/smart_compose/requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir -r /code/requirements.txt

# copy rest of the code
COPY ./src/back_end/ /code/src/back_end/
COPY ./data/smart_compose/train.pickle/ ./data/smart_compose/train.pickle/

# run api
CMD ["uvicorn", "src.back_end.smart_compose.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8002"]