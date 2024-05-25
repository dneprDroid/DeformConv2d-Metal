# docker build --platform=linux/amd64 --no-cache -t deform-conv2d-test ./packageTest 
# docker run -v $(realpath ./demoApp/generated):/test/output deform-conv2d-test 

FROM python:3.10-slim-bookworm

WORKDIR /test

RUN apt update && apt install -y git 

COPY ./torch-requirements.txt ./
RUN pip install -r torch-requirements.txt

COPY ./other-requirements.txt ./
RUN pip install -r other-requirements.txt

COPY ./convert_test.py ./

CMD [ "python", "convert_test.py" ]