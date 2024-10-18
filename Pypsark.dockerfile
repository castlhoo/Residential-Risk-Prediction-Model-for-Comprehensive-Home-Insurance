# 1. Python 기반 이미지 사용
FROM python:3.9-slim

# 2. PySpark와 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless curl && \
    pip install pyspark

# 3. 워킹 디렉토리 설정
WORKDIR /app

# 4. 로컬의 파이썬 모델 파일과 CSV 파일을 컨테이너 내부로 복사
COPY pyspark_model.py /app/pyspark_model.py
COPY APT_Crime_Data.csv /app/APT_Crime_Data.csv
COPY APT_Prediction.csv /app/APT_Prediction.csv

# 5. 모델 실행에 필요한 추가 Python 패키지 설치 (필요한 경우)
# 예시로 pandas 설치
RUN pip install pandas

# 6. 실행 명령어 설정 (모델 파일 실행)
CMD ["python", "/app/pyspark_model.py"]
