FROM public.ecr.aws/lambda/python:3.13

RUN pip install --no-cache-dir clickhouse-connect==0.8.18

COPY handler.py ${LAMBDA_TASK_ROOT}

CMD ["handler.handler"]