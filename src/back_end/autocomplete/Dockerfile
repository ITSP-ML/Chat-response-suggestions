FROM python:3.9
# 
WORKDIR /code
# export port for nginx
EXPOSE 8000


# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./src/back_end/autocomplete/requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir -r /code/requirements.txt

# copy rest of the code
COPY ./src/back_end/ /code/src/back_end/
COPY ./data/prod_v1/agent_data_3.csv/ ./data/prod_v1/agent_data_3.csv/

# run api
CMD ["uvicorn", "src.back_end.autocomplete.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]