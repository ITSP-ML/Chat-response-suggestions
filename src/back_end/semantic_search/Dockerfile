# FROM continuumio/miniconda3
# #  Expose the port 5000 of the docker container
# # EXPOSE 5008
# WORKDIR /app

# COPY ./environment.lock.yml /code/environment.lock.yml

# RUN conda env update -f /code/environment.lock.yml --prune
# RUN echo "source activate env" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH
# # copy all files into the app folder
# COPY .. /code/app


# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

FROM python:3.9
# 
WORKDIR /code
# export port for nginx
EXPOSE 8001


# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./src/back_end/semantic_search/requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir -r /code/requirements.txt

# copy rest of the code
COPY ./src/back_end/ /code/src/back_end/
COPY ./data/prod_v1/agent_data.csv/ ./data/prod_v1/agent_data.csv/
# copy the data


# "--port", "8010"
CMD ["uvicorn", "src.back_end.semantic_search.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8001"]



