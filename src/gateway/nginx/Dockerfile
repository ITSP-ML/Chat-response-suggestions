FROM nginx

# Override the default nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf
COPY ./src/gateway/nginx/nginx.conf /etc/nginx/nginx.conf